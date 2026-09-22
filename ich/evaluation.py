"""Inference uses the saved extractor, PCA, label order, size, and thresholds."""
import json
from pathlib import Path
import numpy as np
from . import LABELS
from .dataset import make_dataset, validate_manifest, manifest_digest
from .preprocessing import PREPROCESS_VERSION, WINDOWS, load_image
from .training import sha256, write_json
from .metrics import evaluate_predictions


class Predictor:
    def __init__(self, bundle):
        import tensorflow as tf
        from . import models  # registers serialization adapters
        from .transforms import TrainingPCA
        bundle = Path(bundle)
        self.metadata = m = json.loads((bundle/"metadata.json").read_text())
        if m["status"] != "trained" or m["schema"] != 1 or m["labels"] != list(LABELS):
            raise ValueError("Invalid trained bundle/label order")
        if m["preprocessing"] != PREPROCESS_VERSION or m["windows"] != [list(x) for x in WINDOWS]:
            raise ValueError("Preprocessing contract mismatch")
        expected = {"model.keras", "extractor.keras", "pca.npz"} if m["baseline"] == "pca_ensemble" else {"model.keras"}
        if set(m["artifacts"]) != expected:
            raise ValueError("Incomplete bundle")
        for name, digest in m["artifacts"].items():
            if sha256(bundle/name) != digest:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        self.model = tf.keras.models.load_model(bundle/"model.keras", compile=False, safe_mode=True)
        self.extractor = self.pca = None
        if m["baseline"] == "pca_ensemble":
            self.extractor = tf.keras.models.load_model(bundle/"extractor.keras", compile=False, safe_mode=True)
            self.pca = TrainingPCA.load(bundle/"pca.npz")
            if self.pca.split_digest != m["split_digest"]:
                raise ValueError("PCA split provenance mismatch")

    def predict_batch(self, images):
        x = np.asarray(images, dtype=np.float32)
        if x.ndim != 4 or tuple(x.shape[1:]) != (*self.metadata["size"], 3) or not np.isfinite(x).all() or np.any((x < 0) | (x > 255)):
            raise ValueError("Expected preprocessed image batch matching saved contract")
        if self.pca is not None:
            x = self.pca.transform(self.extractor(x, training=False).numpy())
        return self.model(x, training=False).numpy()

    def predict_path(self, path):
        p = self.predict_batch(load_image(path, self.metadata["size"])[None])[0]
        return {label: dict(probability=float(v), positive=bool(v >= t))
                for label, v, t in zip(LABELS, p, self.metadata["thresholds"])}


def evaluate(bundle, manifest, output, partition="test", batch_size=32):
    validate_manifest(manifest, partitioned=True)
    from .dataset import verify_files
    verify_files(manifest)
    if partition not in ("validation", "test"):
        raise ValueError("Evaluation requires a held-out partition")
    predictor = Predictor(bundle)
    if manifest_digest(manifest) != predictor.metadata["split_digest"]:
        raise ValueError("Evaluation manifest differs from training split")
    if predictor.pca is not None and set(predictor.pca.fit_ids) != set(manifest.loc[manifest.partition.eq("train"), "image_id"]):
        raise ValueError("PCA fit IDs differ from training split")
    frame = manifest[manifest.partition.eq(partition)].sort_values("image_id")
    ds = make_dataset(frame, tuple(predictor.metadata["size"]), batch_size)
    predictions = np.concatenate([predictor.predict_batch(x.numpy()) for x, _ in ds])
    report = evaluate_predictions(frame[list(LABELS)].to_numpy(), predictions, predictor.metadata["thresholds"])
    report.update(partition=partition, split_digest=predictor.metadata["split_digest"], patient_count=int(frame.patient_id.nunique()))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = frame[["image_id", "patient_id"]].copy()
    for i, label in enumerate(LABELS):
        rows[f"true_{label}"] = frame[label]
        rows[f"prob_{label}"] = predictions[:, i]
    rows.to_csv(out/"predictions.csv", index=False)
    write_json(out/"metrics.json", report)
    return report
