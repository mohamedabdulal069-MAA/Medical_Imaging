"""PCA fitted only to explicit training image IDs; portable non-pickle state."""
import numpy as np
from .dataset import validate_manifest, manifest_digest


class TrainingPCA:
    def __init__(self, components=512):
        self.n_components = components
        self.mean = self.components = None

    def fit(self, features, image_ids, manifest):
        from sklearn.decomposition import PCA
        validate_manifest(manifest, partitioned=True)
        ids = list(image_ids)
        expected = set(manifest.loc[manifest.partition.eq("train"), "image_id"])
        if len(set(ids)) != len(ids) or set(ids) != expected:
            raise ValueError("PCA fit IDs must be exactly the training partition")
        x = np.asarray(features)
        if x.ndim != 2 or len(x) != len(ids) or not np.isfinite(x).all():
            raise ValueError("Invalid feature matrix")
        if self.mean is not None:
            raise ValueError("PCA already fitted; create a new run to refit")
        if not np.any(np.ptp(x, axis=0) > 0):
            raise ValueError("PCA training features have no variance")
        if not 1 <= self.n_components <= min(x.shape):
            raise ValueError("PCA components exceed training samples/features")
        pca = PCA(n_components=self.n_components, svd_solver="full")
        pca.fit(x)
        self.mean, self.components = pca.mean_, pca.components_
        self.fit_ids = tuple(ids)
        self.split_digest = manifest_digest(manifest)
        return self

    def transform(self, features):
        x = np.asarray(features)
        if self.mean is None:
            raise ValueError("PCA is not fitted")
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Wrong/nonfinite CNN features for PCA")
        return ((x-self.mean) @ self.components.T).astype(np.float32)

    def save(self, path):
        if self.mean is None:
            raise ValueError("Cannot save unfitted PCA")
        np.savez(path, mean=self.mean, components=self.components, fit_ids=np.array(self.fit_ids), split_digest=self.split_digest)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            obj = cls(len(state["components"]))
            obj.mean, obj.components = state["mean"], state["components"]
            obj.fit_ids = tuple(state["fit_ids"].tolist())
            obj.split_digest = str(state["split_digest"])
        return obj
