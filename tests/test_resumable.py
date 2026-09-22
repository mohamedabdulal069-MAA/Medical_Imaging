"""Actual optimizer recovery and no held-out-pixel access on synthetic CTs."""
import json
import numpy as np
import pytest
import tensorflow as tf
from ich import resumable, models
from test_pipeline import synthetic_manifest, tiny_image_model


def test_interrupted_training_matches_continuous_without_dropout(tmp_path, dicom, monkeypatch):
    manifest = synthetic_manifest(tmp_path, dicom)
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)
    monkeypatch.setattr(models, "build_model", tiny_image_model)
    seen = []
    original = resumable.checked_image
    def checked(row, size):
        assert row.partition != "test"
        seen.append(row.image_id)
        return original(row, size)
    monkeypatch.setattr(resumable, "checked_image", checked)
    kwargs = dict(epochs=2, batch_size=2, weights=None, workers=1,
                  require_gpu=False, max_minutes=10, checkpoint_every=2)
    result = resumable.run(path, tmp_path / "resume", max_batches=1, **kwargs)
    assert result["status"] == "paused"
    _, saved = resumable.read_checkpoint(tmp_path / "resume")
    assert saved["optimizer_steps"] == 1
    first_ids = seen.copy()
    seen.clear()
    # Recover exactly the next training batch, not the already committed one.
    resumable.run(path, tmp_path / "resume", max_batches=1, **kwargs)
    assert not set(first_ids) & set(seen)
    paused_validation = resumable.run(path, tmp_path / "resume", max_batches=2, **kwargs)
    assert paused_validation["phase"] == "validation"
    assert paused_validation["validation_n"] > 0
    completed = resumable.run(path, tmp_path / "resume", **kwargs)
    assert completed["status"] == "complete"
    resumed = tf.keras.models.load_model(tmp_path / "resume/model.keras")
    resumable.run(path, tmp_path / "continuous", **kwargs)
    continuous = tf.keras.models.load_model(tmp_path / "continuous/model.keras")
    assert int(resumed.optimizer.iterations.numpy()) == int(continuous.optimizer.iterations.numpy())
    for a, b in zip(resumed.weights, continuous.weights):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6, atol=1e-6)
    assert len(completed["history"]) == 2
    with pytest.raises(ValueError, match="contract changed"):
        resumable.run(path, tmp_path / "resume", **dict(kwargs, seed=17))
    # Resumption refuses a changed checkpoint, rather than silently starting over.
    folder, _ = resumable.read_checkpoint(tmp_path / "resume")
    with (folder / "model.keras").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        resumable.run(path, tmp_path / "resume", **kwargs)


def test_failure_during_checkpoint_leaves_previous_recoverable(tmp_path, monkeypatch):
    model = tiny_image_model()
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.train_on_batch(np.zeros((1, 224, 224, 3), np.float32), np.zeros((1, 6), np.float32))
    state = dict(epoch=0, phase="train", next_batch=1)
    resumable.save_checkpoint(model, state, tmp_path)
    pointer = (tmp_path / "current.json").read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated failed model write")
    monkeypatch.setattr(model, "save", fail)
    with pytest.raises(OSError):
        resumable.save_checkpoint(model, dict(state, next_batch=2), tmp_path)
    assert (tmp_path / "current.json").read_bytes() == pointer
    _, recovered = resumable.read_checkpoint(tmp_path)
    assert recovered["next_batch"] == 1


def test_lazy_check_rejects_stale_pixels(tmp_path, dicom):
    frame = synthetic_manifest(tmp_path, dicom)
    frame.loc[0, "pixel_hash"] = "changed"
    with pytest.raises(ValueError, match="pixels changed"):
        resumable.checked_image(next(frame.itertuples()), (224, 224))
