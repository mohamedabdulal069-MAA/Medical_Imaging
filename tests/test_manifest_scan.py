import json
import numpy as np
import pytest
from ich import LABELS
from ich import manifest_scan as scan


def make_data(tmp_path, dicom, broken=False):
    folder = tmp_path/"dicoms"
    folder.mkdir()
    rows = ["ID,Label"]
    for i in range(10):
        dicom.PatientID = f"p{i}"
        dicom.StudyInstanceUID = f"1.2.826.{i+1}"
        dicom.SeriesInstanceUID = f"1.2.826.{i+1}.1"
        dicom.SOPInstanceUID = f"1.2.826.{i+1}.1.1"
        dicom.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        dicom.file_meta.MediaStorageSOPClassUID = dicom.SOPClassUID
        dicom.file_meta.MediaStorageSOPInstanceUID = dicom.SOPInstanceUID
        dicom.PixelData = np.array([[500+i,510+i],[520+i,530+i]], dtype="<i2").tobytes()
        if broken and i == 9:
            dicom.PixelData = b"\x00\x00"
        dicom.save_as(folder/f"ID_{i}.dcm", enforce_file_format=True)
        for label in LABELS:
            rows.append(f"ID_{i}_{label},{int(i%2 == 0 and label in ('any','epidural'))}")
    path = tmp_path/"labels.csv"
    path.write_text("\n".join(rows))
    return path, folder


def test_scan_cache_resume_and_changed_file(tmp_path, dicom, monkeypatch):
    labels, folder = make_data(tmp_path, dicom)
    checkpoint = tmp_path/"checkpoint"
    result = scan.prepare_resumable(labels, folder, tmp_path/"first.csv", checkpoint)
    assert len(result) == 10
    original = scan.scan_one
    calls = []
    def counted(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(scan, "scan_one", counted)
    scan.prepare_resumable(labels, folder, tmp_path/"second.csv", checkpoint)
    assert calls == []
    with (folder/"ID_0.dcm").open("ab") as stream:
        stream.write(b"\x00\x00")
    scan.prepare_resumable(labels, folder, tmp_path/"third.csv", checkpoint)
    assert calls == ["ID_0"]


def test_unknown_errors_block_even_with_known_exclusion_enabled(tmp_path, dicom):
    labels, folder = make_data(tmp_path, dicom, broken=True)
    checkpoint = tmp_path/"checkpoint"
    with pytest.raises(ValueError, match="unresolved"):
        scan.prepare_resumable(labels, folder, tmp_path/"manifest.csv", checkpoint, exclude_confirmed_short_payload=True)
    assert not (tmp_path/"manifest.csv").exists()
    errors = json.loads((checkpoint/"unresolved-errors.json").read_text())
    assert errors[0]["image_id"] == "ID_9"
    assert json.loads((checkpoint/"exclusions.json").read_text()) == []


def test_known_exclusion_requires_exact_observed_defect(tmp_path, dicom, monkeypatch):
    path = tmp_path/"ID_6431af929.dcm"
    path.write_bytes(bytes(154410))
    dicom.Rows = dicom.Columns = 512
    dicom.PatientID = "p1"
    dicom.StudyInstanceUID = "1.2.3"
    dicom.SeriesInstanceUID = "1.2.3.1"
    dicom.SOPInstanceUID = "1.2.3.1.1"
    dicom.PixelData = bytes(153710)
    monkeypatch.setattr(scan.pydicom, "dcmread", lambda _: dicom)
    record = scan.scan_one("ID_6431af929", path)
    assert record["status"] == "known_short_payload"
    assert record["file_sha256"] == scan.file_hash(path)
    assert scan.scan_one("ID_other", path)["status"] == "error"
    dicom.PixelData = bytes(524288)
    assert scan.scan_one("ID_6431af929", path)["status"] == "ok"


def test_interrupt_preserves_completed_work(tmp_path, dicom, monkeypatch):
    labels, folder = make_data(tmp_path, dicom)
    checkpoint = tmp_path/"checkpoint"
    original = scan.scan_one
    calls = []
    def interrupted(image_id, path):
        calls.append(image_id)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return original(image_id, path)
    monkeypatch.setattr(scan, "scan_one", interrupted)
    with pytest.raises(KeyboardInterrupt):
        scan.prepare_resumable(labels, folder, tmp_path/"manifest.csv", checkpoint)
    resumed = []
    def counted(image_id, path):
        resumed.append(image_id)
        return original(image_id, path)
    monkeypatch.setattr(scan, "scan_one", counted)
    scan.prepare_resumable(labels, folder, tmp_path/"manifest.csv", checkpoint)
    assert len(resumed) == 7
