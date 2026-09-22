"""Checkpointed preparation for immutable datasets, with explicit exclusion evidence."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import warnings
import numpy as np
import pandas as pd
import pydicom
from ich.dataset import read_labels, split_manifest
from ich import preprocessing


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_one(image_id, path):
    """All failures retain their image ID; only the known payload defect is eligible."""
    record = {"image_id": image_id, "path": str(path), "status": "error", "uid_warnings": 0}
    captured = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", UserWarning)
            ds = pydicom.dcmread(path)
            for key, tag in zip(("patient_id", "study_id", "series_id", "sop_id"),
                                ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")):
                record[key] = str(getattr(ds, tag, "")).strip()
                if not record[key]:
                    raise ValueError(f"Missing identity: {tag}")
            payload = len(ds.get("PixelData", b""))
            record["payload_bytes"] = payload
            # An exact fingerprint of the defect observed on Kaggle, not a general skip rule.
            signature = (path.stat().st_size, int(ds.Rows), int(ds.Columns),
                int(getattr(ds, "NumberOfFrames", 1)), int(ds.SamplesPerPixel), int(ds.BitsAllocated),
                int(ds.BitsStored), int(ds.PixelRepresentation), str(ds.file_meta.TransferSyntaxUID), payload)
            known = image_id == "ID_6431af929" and signature == (
                154410, 512, 512, 1, 1, 16, 16, 1, "1.2.840.10008.1.2.1", 153710)
            try:
                hu = np.ascontiguousarray(preprocessing.dicom_to_hu(ds), dtype="<f4")
            except ValueError as error:
                if known and "number of bytes of pixel data is less than expected" in str(error):
                    record.update(status="known_short_payload", error=str(error),
                                  file_sha256=file_hash(path), expected_payload_bytes=524288)
                    return record
                raise
            record.update(status="ok", pixel_hash=hashlib.sha256(str(hu.shape).encode()+hu.tobytes()).hexdigest())
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["uid_warnings"] = sum("Invalid value for VR UI:" in str(w.message) for w in captured)
        other = [str(w.message) for w in captured if "Invalid value for VR UI:" not in str(w.message)]
        if other:
            record["other_warnings"] = other
    return record


def prepare_resumable(labels_csv, dicom_dir, output, checkpoint_dir, seed=42,
                      exclude_confirmed_short_payload=False, progress_every=1000):
    """Resume unchanged, immutable input mounts. Unexpected errors block final splitting.

    Preserve checkpoint_dir between sessions. Source location, labels, reader code,
    pydicom/numpy versions and per-file size/mtime are bound to the cache. For mutable
    datasets use a new checkpoint directory; size/mtime is not a content proof.
    """
    output, checkpoint_dir = Path(output), Path(checkpoint_dir)
    dicom_dir = Path(dicom_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Existing manifest protected: {output}")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    labels = read_labels(labels_csv)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=1, dicom_dir=str(dicom_dir), labels_sha256=file_hash(labels_csv),
                    preprocessing_sha256=file_hash(preprocessing.__file__),
                    scanner_version="short-payload-review-v1",
                    pydicom=pydicom.__version__, numpy=np.__version__)
    contract_json = json.dumps(contract, sort_keys=True)
    db = sqlite3.connect(checkpoint_dir/"scan.sqlite")
    started = time.monotonic()
    scanned = cached = 0
    records = []
    print(f"Checking {len(labels):,} images. Checkpoint: {checkpoint_dir}", flush=True)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, signature TEXT, record TEXT)")
        previous = db.execute("SELECT value FROM config").fetchone()
        if previous and previous[0] != contract_json:
            raise ValueError("Checkpoint source/reader contract changed; use a new checkpoint directory")
        if not previous:
            db.execute("INSERT INTO config VALUES (?)", (contract_json,))
            db.commit()
        for index, image_id in enumerate(labels.image_id, start=1):
            path = dicom_dir/f"{image_id}.dcm"
            stat = path.stat() if path.exists() else None
            signature = json.dumps([stat.st_size, stat.st_mtime_ns] if stat else None)
            previous = db.execute("SELECT signature, record FROM images WHERE id=?", (image_id,)).fetchone()
            if previous and previous[0] == signature:
                record = json.loads(previous[1])
                cached += 1
            else:
                record = scan_one(image_id, path)
                db.execute("INSERT OR REPLACE INTO images VALUES (?, ?, ?)", (image_id, signature, json.dumps(record)))
                scanned += 1
            records.append(record)
            if index % 100 == 0:
                db.commit()
            if index % progress_every == 0 or index == len(labels):
                elapsed = (time.monotonic()-started)/60
                print(f"{index:,}/{len(labels):,} checked | {cached:,} cached | {scanned:,} read | {elapsed:.1f} min", flush=True)
    finally:
        db.commit()
        db.close()
    accepted = [r for r in records if r["status"] == "known_short_payload" and exclude_confirmed_short_payload]
    errors = [r for r in records if r["status"] != "ok" and r not in accepted]
    # Always write inspection evidence before attempting the final split.
    for name, rows in (("exclusions.json", accepted), ("unresolved-errors.json", errors)):
        (checkpoint_dir/name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    warning_records = [{"image_id": r["image_id"], "uid_warnings": r["uid_warnings"],
                        "other_warnings": r.get("other_warnings", [])}
                       for r in records if r["uid_warnings"] or r.get("other_warnings")]
    with (checkpoint_dir/"warnings.jsonl").open("w", encoding="utf-8") as stream:
        for record in warning_records:
            stream.write(json.dumps(record)+"\n")
    summary = dict(total_images=len(labels), readable=sum(r["status"] == "ok" for r in records),
                   excluded=len(accepted), unresolved=len(errors), seed=seed,
                   uid_warning_count=sum(r["uid_warnings"] for r in records), contract=contract)
    (checkpoint_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if errors:
        print(json.dumps(errors[:10], indent=2), flush=True)
        raise ValueError(f"{len(errors)} unresolved image errors; see unresolved-errors.json. No manifest created.")
    kept = pd.DataFrame([{key: r[key] for key in (
        "image_id", "path", "patient_id", "study_id", "series_id", "sop_id", "pixel_hash")}
        for r in records if r["status"] == "ok"])
    frame = kept.merge(labels, on="image_id", validate="one_to_one")
    excluded_labels = labels[labels.image_id.isin([r["image_id"] for r in accepted])]
    excluded_labels.to_csv(checkpoint_dir/"excluded-labels.csv", index=False)
    frame = split_manifest(frame, seed=seed)  # All existing identity/leakage checks remain mandatory.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(frame.groupby("partition").agg(images=("image_id", "size"), patients=("patient_id", "nunique")), flush=True)
    print(f"Saved {output}; documented exclusions: {len(accepted)}", flush=True)
    return frame
