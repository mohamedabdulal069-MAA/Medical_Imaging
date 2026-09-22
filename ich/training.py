"""Reproducible training and self-contained inference bundles."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import numpy as np
from . import LABELS
from .dataset import validate_manifest, manifest_digest, make_dataset
from .preprocessing import PREPROCESS_VERSION, WINDOWS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed):
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def train(manifest, output, baseline="resnet50", epochs=3, batch_size=32, seed=42,
          components=512, weights="imagenet", weighted_loss=False):
    validate_manifest(manifest, partitioned=True)
    from .dataset import verify_files
    verify_files(manifest)
    if epochs < 1 or batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    set_seed(seed)
    import tensorflow as tf
    from .models import build_model, build_pca_head, feature_model, image_size, PCA_BRANCHES, BASELINES
    from .losses import binary_loss, training_positive_weights
    from .transforms import TrainingPCA
    if baseline not in BASELINES:
        raise ValueError("Unknown baseline")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    manifest.to_csv(output/"manifest.csv", index=False)
    frames = {p: manifest[manifest.partition.eq(p)] for p in ("train", "validation")}
    size = image_size(baseline)
    datasets = {p: make_dataset(f, size, batch_size, training=(p == "train" and baseline != "pca_ensemble"), seed=seed)
                for p, f in frames.items()}
    pca = extractor = None
    if baseline == "pca_ensemble":
        extractor = feature_model(PCA_BRANCHES, size, weights)
        features = {p: extractor.predict(ds, verbose=0) for p, ds in datasets.items()}
        pca = TrainingPCA(components).fit(features["train"], frames["train"].image_id, manifest)
        # Only frozen CNN features are cached; held-out data never enters fit.
        datasets = {p: tf.data.Dataset.from_tensor_slices((pca.transform(features[p]), f[list(LABELS)].to_numpy(np.float32)))
                    for p, f in frames.items()}
        datasets["train"] = datasets["train"].shuffle(len(frames["train"]), seed=seed)
        datasets = {p: ds.batch(batch_size) for p, ds in datasets.items()}
        model = build_pca_head(components)
    else:
        model = build_model(baseline, weights)
    positive_weights = training_positive_weights(frames["train"]) if weighted_loss else None
    lr = 1e-3 if baseline in ("densenet121", "pca_ensemble") else 1e-4
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=binary_loss(positive_weights),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")])
    history = model.fit(datasets["train"], validation_data=datasets["validation"], epochs=epochs, shuffle=False,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)])
    if not history.epoch or any(not np.isfinite(v).all() for v in history.history.values()):
        raise RuntimeError("Training did not finish with finite history; no bundle saved")
    # The successful completion marker is written last, after trained weights.
    model.save(output/"model.keras")
    artifacts = ["model.keras"]
    if pca is not None:
        pca.save(output/"pca.npz")
        extractor.save(output/"extractor.keras")
        artifacts += ["pca.npz", "extractor.keras"]
    write_json(output/"history.json", history.history)
    versions = {p: importlib.metadata.version(p) for p in ("tensorflow", "keras", "numpy", "pandas", "pydicom", "scikit-learn", "scikit-image")}
    metadata = dict(schema=1, status="trained", baseline=baseline, labels=list(LABELS), size=list(size),
        preprocessing=PREPROCESS_VERSION, windows=WINDOWS, thresholds=[0.5]*6,
        threshold_policy="fixed a priori; no test tuning", seed=seed, epochs_requested=epochs,
        epochs_completed=len(history.epoch), batch_size=batch_size, learning_rate=lr,
        initial_weights=weights, loss_positive_weights=None if positive_weights is None else positive_weights.tolist(),
        split_digest=manifest_digest(manifest), versions=versions, python=platform.python_version(),
        platform=platform.platform(), devices=[str(x) for x in tf.config.list_physical_devices()],
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")},
        artifacts={p: sha256(output/p) for p in artifacts})
    write_json(output/"metadata.json", metadata)
    return metadata
