"""Synthetic plumbing checks, never clinical experiments."""
import json
import numpy as np
import pytest
import tensorflow as tf
from ich.dataset import build_manifest, split_manifest, make_dataset, verify_files
from ich.training import train
from ich.evaluation import Predictor, evaluate
from ich import LABELS


def synthetic_manifest(tmp_path, dicom):
    folder = tmp_path/"dicoms"
    folder.mkdir()
    rows = ["ID,Label"]
    for i in range(8):
        dicom.PatientID = f"p{i}"
        dicom.StudyInstanceUID = f"1.2.826.{i+1}"
        dicom.SeriesInstanceUID = f"1.2.826.{i+1}.1"
        dicom.SOPInstanceUID = f"1.2.826.{i+1}.1.1"
        dicom.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        dicom.file_meta.MediaStorageSOPClassUID = dicom.SOPClassUID
        dicom.file_meta.MediaStorageSOPInstanceUID = dicom.SOPInstanceUID
        dicom.PixelData = np.array([[500+i, 510+i], [530+i, 540+i]], dtype="<i2").tobytes()
        dicom.save_as(folder/f"ID_{i}.dcm", enforce_file_format=True)
        for label in LABELS:
            rows.append(f"ID_{i}_{label},{int(i%2==0 and label in ('epidural','any'))}")
    labels = tmp_path/"labels.csv"
    labels.write_text("\n".join(rows))
    return split_manifest(build_manifest(labels, folder))


def tiny_image_model(*args, **kwargs):
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Rescaling(1/255)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(6, activation="sigmoid")(x))


def tiny_extractor(*args, **kwargs):
    inputs = tf.keras.Input((224, 224, 3))
    return tf.keras.Model(inputs, tf.keras.layers.GlobalAveragePooling2D()(inputs))


@pytest.mark.parametrize("baseline", ["resnet50", "pca_ensemble"])
def test_train_save_reload_evaluate_pipeline(tmp_path, dicom, monkeypatch, baseline):
    from ich import models
    monkeypatch.setattr(models, "build_model", tiny_image_model)
    monkeypatch.setattr(models, "feature_model", tiny_extractor)
    manifest = synthetic_manifest(tmp_path, dicom)
    bundle = tmp_path/"run"
    # A save before any optimizer update is a regression even if reload succeeds.
    original_save = tf.keras.Model.save
    def checked_save(self, *args, **kwargs):
        if getattr(self, "optimizer", None) is not None:
            assert int(self.optimizer.iterations.numpy()) > 0
        return original_save(self, *args, **kwargs)
    monkeypatch.setattr(tf.keras.Model, "save", checked_save)
    train(manifest, bundle, baseline=baseline, epochs=2, batch_size=2, weights=None, components=2)
    metadata = json.loads((bundle/"metadata.json").read_text())
    assert metadata["epochs_completed"] == 2
    predictor = Predictor(bundle)
    row = manifest[manifest.partition.eq("test")].iloc[0]
    path_prediction = predictor.predict_path(row.path)
    batch = next(iter(make_dataset(manifest[manifest.image_id.eq(row.image_id)], (224,224))))[0]
    values = predictor.predict_batch(batch.numpy())[0]
    np.testing.assert_allclose(values, [path_prediction[l]["probability"] for l in LABELS])
    report = evaluate(bundle, manifest, tmp_path/"evaluation")
    assert report["n"] == sum(manifest.partition.eq("test"))
    changed = manifest.copy()
    changed.loc[changed.partition.eq("test"), "partition"] = "validation"
    with pytest.raises(ValueError):
        evaluate(bundle, changed, tmp_path/"wrong")
    with open(bundle/"model.keras", "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(bundle)


def test_stale_dicom_manifest_rejected(tmp_path, dicom):
    manifest = synthetic_manifest(tmp_path, dicom)
    manifest.loc[0, "pixel_hash"] = "wrong"
    with pytest.raises(ValueError, match="pixels changed"):
        verify_files(manifest)
