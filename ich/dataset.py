"""One record per image, explicit identities, and immutable patient partitions."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from . import LABELS

IDENTITIES = ("patient_id", "study_id", "series_id", "sop_id", "image_id", "pixel_hash")
PARTITIONS = ("train", "validation", "test")


def read_labels(path):
    df = pd.read_csv(path)
    parts = df.ID.str.rsplit("_", n=1, expand=True)
    df = df.assign(image_id=parts[0], subtype=parts[1])
    if not df.subtype.isin(LABELS).all() or not df.Label.isin([0, 1]).all():
        raise ValueError("Unknown subtype or missing/nonbinary label")
    if df.groupby(["image_id", "subtype"]).Label.nunique().gt(1).any():
        raise ValueError("Conflicting duplicate labels")
    wide = df.drop_duplicates(["image_id", "subtype"]).pivot(index="image_id", columns="subtype", values="Label")
    wide = wide.reindex(columns=LABELS)
    validate_labels(wide.to_numpy())
    return wide.reset_index()


def validate_labels(y):
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 6 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected nonempty N x 6 complete binary labels")
    if not np.array_equal(y[:, 5], np.max(y[:, :5], axis=1)):
        raise ValueError("'any' must equal the union of five subtypes")


def build_manifest(labels_csv, dicom_dir):
    import pydicom
    from .preprocessing import dicom_to_hu
    labels = read_labels(labels_csv)
    records = []
    for image_id in labels.image_id:
        path = (Path(dicom_dir) / f"{image_id}.dcm").resolve()
        ds = pydicom.dcmread(path)
        attrs = ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
        values = [str(getattr(ds, a, "")).strip() for a in attrs]
        if not all(values):
            raise ValueError(f"Missing patient/study/series/SOP identity: {image_id}")
        hu = np.ascontiguousarray(dicom_to_hu(ds), dtype="<f4")
        digest = hashlib.sha256(str(hu.shape).encode() + hu.tobytes()).hexdigest()
        records.append(dict(zip(IDENTITIES[:4], values), image_id=image_id, pixel_hash=digest, path=str(path)))
    result = pd.DataFrame(records).merge(labels, on="image_id", validate="one_to_one")
    validate_manifest(result)
    return result


def validate_manifest(df, partitioned=False):
    for col in (*IDENTITIES, "path"):
        if col not in df or df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identity/path: {col}")
    validate_labels(df[list(LABELS)].to_numpy())
    for col in ("image_id", "sop_id", "path"):
        if df[col].duplicated().any():
            raise ValueError(f"Duplicate image: {col}")
    identities = list(IDENTITIES)
    if "split_group" in df:
        if df.split_group.isna().any() or df.split_group.astype(str).str.strip().eq("").any():
            raise ValueError("Missing split group")
        for col in ("patient_id", "pixel_hash"):
            if df.groupby(col).split_group.nunique().gt(1).any():
                raise ValueError(f"Identity spans split groups: {col}")
        identities.append("split_group")
    # Duplicate-linked IDs may share a group without asserting they are one patient.
    strict = ("study_id", "series_id") if "split_group" in df else ("study_id", "series_id", "pixel_hash")
    for col in strict:
        if df.groupby(col).patient_id.nunique().gt(1).any():
            raise ValueError(f"Identity/copy shared across patients: {col}")
    if partitioned:
        if "partition" not in df or set(df.partition) != set(PARTITIONS):
            raise ValueError("All three nonempty partitions are required")
        for col in identities:
            if df.groupby(col).partition.nunique().gt(1).any():
                raise ValueError(f"Leakage across partitions: {col}")


def split_manifest(df, seed=42, validation=0.15, test=0.15):
    validate_manifest(df)
    if validation <= 0 or test <= 0 or validation + test >= 1:
        raise ValueError("Invalid split fractions")
    group_column = "split_group" if "split_group" in df else "patient_id"
    patients = np.array(sorted(df[group_column].unique()))
    np.random.default_rng(seed).shuffle(patients)
    nv, nt = max(1, round(len(patients)*validation)), max(1, round(len(patients)*test))
    if nv + nt >= len(patients):
        raise ValueError("Too few patients for requested split")
    lookup = {p: "validation" if i < nv else "test" if i < nv+nt else "train" for i, p in enumerate(patients)}
    result = df.copy().sort_values("image_id").reset_index(drop=True)
    result["partition"] = result[group_column].map(lookup)
    validate_manifest(result, partitioned=True)
    return result


def manifest_digest(df):
    cols = [*IDENTITIES, *LABELS, "partition"]
    if "split_group" in df:
        cols.append("split_group")
    return hashlib.sha256(df.sort_values("image_id")[cols].to_csv(index=False).encode()).hexdigest()


def load_manifest(path):
    df = pd.read_csv(path, dtype={c: str for c in (*IDENTITIES, "path", "partition", "split_group")}, keep_default_na=False)
    validate_manifest(df, partitioned=True)
    return df


def verify_files(df):
    """Detect stale or edited paths, identities, and pixels before a run."""
    import pydicom
    from .preprocessing import dicom_to_hu
    for row in df.itertuples():
        ds = pydicom.dcmread(row.path)
        actual = tuple(str(getattr(ds, a, "")).strip() for a in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"))
        if actual != (row.patient_id, row.study_id, row.series_id, row.sop_id):
            raise ValueError(f"DICOM identity changed: {row.image_id}")
        hu = np.ascontiguousarray(dicom_to_hu(ds), dtype="<f4")
        digest = hashlib.sha256(str(hu.shape).encode() + hu.tobytes()).hexdigest()
        if digest != row.pixel_hash:
            raise ValueError(f"DICOM pixels changed: {row.image_id}")


def make_dataset(frame, size, batch_size=32, training=False, seed=42):
    import tensorflow as tf
    from .preprocessing import load_image
    frame = frame.sort_values("image_id").reset_index(drop=True)
    paths, labels = frame.path.to_numpy(), frame[list(LABELS)].to_numpy(dtype=np.float32)
    def generate():
        for path, label in zip(paths, labels):
            yield load_image(path, size), label
    ds = tf.data.Dataset.from_generator(generate, output_signature=(tf.TensorSpec((*size, 3), tf.float32), tf.TensorSpec((6,), tf.float32)))
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(frame)))
    if training:
        ds = ds.shuffle(min(len(frame), 2048), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return ds.batch(batch_size).with_options(options).prefetch(1)
