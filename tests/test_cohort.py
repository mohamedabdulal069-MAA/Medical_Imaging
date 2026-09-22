import json
import sqlite3
import hashlib
import pandas as pd
import pytest
from ich import LABELS, preprocessing
from ich.cohort import resolve_cohort, finalize_checkpoint
from ich.dataset import validate_manifest, manifest_digest


def linked_frame(manifest):
    frame = manifest.drop(columns="partition").copy()
    # Two discordant pairs create A-B-C linkage even though B loses all its images.
    frame.loc[frame.image_id.isin(["0_0", "1_0"]), "pixel_hash"] = "conflict_ab"
    frame.loc[frame.image_id.isin(["1_1", "2_0"]), "pixel_hash"] = "conflict_bc"
    frame.loc[frame.image_id.isin(["4_0", "6_0"]), "pixel_hash"] = "agree_de"
    return frame


def test_quarantine_dedup_and_transitive_grouping(manifest):
    frame = linked_frame(manifest)
    with pytest.raises(ValueError, match="explicit quarantine"):
        resolve_cohort(frame)
    result, exclusions, mapping = resolve_cohort(frame, quarantine_conflicts=True)
    assert set(exclusions.image_id) == {"0_0", "1_0", "1_1", "2_0", "6_0"}
    assert len(result) + len(exclusions) == len(frame)
    assert not result.pixel_hash.duplicated().any()
    assert result[result.patient_id.isin(["p0", "p2"])].partition.nunique() == 1
    assert mapping[mapping.patient_id.isin(["p0", "p1", "p2"])].split_group.nunique() == 1
    assert result[result.patient_id.isin(["p4", "p6"])].partition.nunique() == 1
    assert exclusions.set_index("image_id").loc["6_0", "retained_image_id"] == "4_0"
    validate_manifest(result, partitioned=True)
    reordered, _, _ = resolve_cohort(frame.sample(frac=1, random_state=7), quarantine_conflicts=True)
    assert manifest_digest(result) == manifest_digest(reordered)


def test_linked_ids_cannot_cross_partitions(manifest):
    result, _, _ = resolve_cohort(linked_frame(manifest), quarantine_conflicts=True)
    index = result.index[result.patient_id.eq("p0")][0]
    old = result.loc[index, "partition"]
    result.loc[index, "partition"] = "test" if old != "test" else "train"
    with pytest.raises(ValueError, match="split_group"):
        validate_manifest(result, partitioned=True)


def test_finalize_complete_cache_without_reading_dicoms(manifest, tmp_path):
    frame = linked_frame(manifest)
    labels_path = tmp_path/"labels.csv"
    long = [{"ID": f"{r.image_id}_{label}", "Label": getattr(r, label)} for r in frame.itertuples() for label in LABELS]
    pd.DataFrame(long).to_csv(labels_path, index=False)
    contract = {"labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
                "preprocessing_sha256": hashlib.sha256(open(preprocessing.__file__, "rb").read()).hexdigest()}
    with sqlite3.connect(tmp_path/"scan.sqlite") as db:
        db.execute("CREATE TABLE config(value TEXT)")
        db.execute("INSERT INTO config VALUES (?)", (json.dumps(contract),))
        db.execute("CREATE TABLE images(id TEXT PRIMARY KEY, record TEXT)")
        for r in frame.to_dict("records"):
            r["status"] = "ok"
            db.execute("INSERT INTO images VALUES (?,?)", (r["image_id"], json.dumps(r)))
    result = finalize_checkpoint(tmp_path, labels_path, tmp_path/"cohort", quarantine_conflicts=True)
    assert len(result) == 35
    assert (tmp_path/"cohort/image-exclusions.csv").exists()
    with sqlite3.connect(tmp_path/"scan.sqlite") as db:
        db.execute("DELETE FROM images WHERE id='0_0'")
    with pytest.raises(ValueError, match="exactly all"):
        finalize_checkpoint(tmp_path, labels_path, tmp_path/"incomplete", quarantine_conflicts=True)
