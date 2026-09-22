"""Explicit conflict quarantine and exact-image deduplication before grouped splitting."""
import hashlib
import json
from pathlib import Path
import sqlite3
import pandas as pd
from ich import LABELS, preprocessing
from ich.dataset import read_labels, validate_manifest, split_manifest


def resolve_cohort(frame, *, quarantine_conflicts=False, seed=42):
    frame = frame.copy().sort_values("image_id").reset_index(drop=True)
    # Build connected components BEFORE removing any images, including conflicted ones.
    parent = {p: p for p in frame.patient_id.unique()}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    repeated = frame[frame.pixel_hash.duplicated(keep=False)]
    for _, rows in repeated.groupby("pixel_hash", sort=True):
        patients = sorted(rows.patient_id.unique())
        for p in patients[1:]:
            a, b = find(patients[0]), find(p)
            parent[max(a, b)] = min(a, b)
    frame["split_group"] = frame.patient_id.map(lambda p: find(p))
    validate_manifest(frame)
    discordant = repeated.groupby("pixel_hash")[list(LABELS)].nunique().gt(1).any(axis=1)
    conflicts = set(discordant.index[discordant])
    if conflicts and not quarantine_conflicts:
        raise ValueError(f"{len(conflicts)} conflicting pixel groups require explicit quarantine policy")
    excluded = frame[frame.pixel_hash.isin(conflicts)].copy()
    excluded["reason"] = "identical_HU_pixels_conflicting_labels_all_copies_quarantined"
    excluded["retained_image_id"] = ""
    consistent = frame[~frame.pixel_hash.isin(conflicts)].copy()
    retained_ids = consistent.groupby("pixel_hash").image_id.first()
    duplicates = consistent[consistent.pixel_hash.duplicated(keep="first")].copy()
    duplicates["reason"] = "identical_HU_pixels_agreeing_labels_redundant_copy"
    duplicates["retained_image_id"] = duplicates.pixel_hash.map(retained_ids)
    exclusions = pd.concat([excluded, duplicates], ignore_index=True)
    retained = consistent.drop_duplicates("pixel_hash", keep="first").copy()
    result = split_manifest(retained, seed=seed)
    mapping = frame[["patient_id", "split_group"]].drop_duplicates().sort_values("patient_id")
    assignments = result[["split_group", "partition"]].drop_duplicates()
    mapping = mapping.merge(assignments, on="split_group", how="left", validate="many_to_one")
    mapping["partition"] = mapping.partition.fillna("excluded_entire_group")
    return result, exclusions, mapping


def finalize_checkpoint(checkpoint_dir, labels_csv, output_dir, *, quarantine_conflicts=False, seed=42):
    """No pixel reread. The checkpoint must be complete and from the same immutable input."""
    checkpoint_dir, output_dir = Path(checkpoint_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Existing cohort output protected: {output_dir}")
    labels = read_labels(labels_csv)
    uri = (checkpoint_dir/"scan.sqlite").resolve().as_uri()+"?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Checkpoint integrity check failed")
        contract = json.loads(db.execute("SELECT value FROM config").fetchone()[0])
        for key, path in (("labels_sha256", labels_csv), ("preprocessing_sha256", preprocessing.__file__)):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != contract[key]:
                raise ValueError(f"Checkpoint contract mismatch: {key}")
        records = []
        unreadable = []
        all_ids = set()
        for image_id, raw in db.execute("SELECT id, record FROM images"):
            all_ids.add(image_id)
            r = json.loads(raw)
            if r["image_id"] != image_id:
                raise ValueError("Checkpoint ID mismatch")
            if r["status"] == "ok":
                records.append({k: r[k] for k in ("image_id", "patient_id", "study_id", "series_id", "sop_id", "pixel_hash", "path")})
            else:
                unreadable.append(r)
    if all_ids != set(labels.image_id):
        raise ValueError("Checkpoint does not contain exactly all labeled images")
    accepted_path = checkpoint_dir/"exclusions.json"
    accepted = json.loads(accepted_path.read_text()) if accepted_path.exists() else []
    if sorted(unreadable, key=lambda r:r["image_id"]) != sorted(accepted, key=lambda r:r["image_id"]):
        raise ValueError("Unreadable image records differ from previously recorded exclusions")
    frame = pd.DataFrame(records).merge(labels, on="image_id", validate="one_to_one")
    result, excluded, mapping = resolve_cohort(frame, quarantine_conflicts=quarantine_conflicts, seed=seed)
    output_dir.mkdir(parents=True)
    result.to_csv(output_dir/"manifest.csv", index=False)
    excluded.to_csv(output_dir/"image-exclusions.csv", index=False)
    mapping.to_csv(output_dir/"patient-group-map.csv", index=False)
    # Pair original annotation rows with unreadable-image evidence as well.
    labels[labels.image_id.isin([r["image_id"] for r in unreadable])].to_csv(output_dir/"unreadable-labels.csv", index=False)
    (output_dir/"unreadable-exclusions.json").write_text(json.dumps(unreadable, indent=2), encoding="utf-8")
    counts = result.groupby("partition").agg(images=("image_id", "size"), patient_ids=("patient_id", "nunique"), split_groups=("split_group", "nunique"))
    counts.to_csv(output_dir/"partition-counts.csv")
    result.groupby("partition")[list(LABELS)].sum().to_csv(output_dir/"positive-label-counts.csv")
    report = dict(policy="quarantine-all-conflicting-pixels;deduplicate-concordant-pixels;group-linked-patient-IDs-v1",
                  seed=seed, labeled_images=len(labels), readable_images=len(frame), unreadable_images=len(unreadable),
                  retained_images=len(result), exclusion_counts=excluded.reason.value_counts().to_dict(),
                  checkpoint_contract=contract)
    (output_dir/"cohort-policy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(counts.to_string(), flush=True)
    print(f"Saved {output_dir/'manifest.csv'}", flush=True)
    return result
