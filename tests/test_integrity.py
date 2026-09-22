import numpy as np
import pytest
from ich.dataset import validate_manifest, split_manifest, read_labels, manifest_digest
from ich.transforms import TrainingPCA
from ich.preprocessing import dicom_to_hu, window_image
from ich.metrics import evaluate_predictions
from ich.losses import training_positive_weights


def test_patient_partitions_are_disjoint_and_reproducible(manifest):
    validate_manifest(manifest, partitioned=True)
    groups = [set(manifest.loc[manifest.partition.eq(p), "patient_id"]) for p in ("train", "validation", "test")]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    shuffled = split_manifest(manifest.sample(frac=1, random_state=10).drop(columns="partition"))
    assert manifest_digest(shuffled) == manifest_digest(manifest)


@pytest.mark.parametrize("identity", ["patient_id", "study_id", "series_id", "sop_id", "image_id", "pixel_hash"])
def test_cross_partition_leakage_rejected(manifest, identity):
    train = manifest.index[manifest.partition.eq("train")][0]
    test = manifest.index[manifest.partition.eq("test")][0]
    manifest.loc[test, identity] = manifest.loc[train, identity]
    with pytest.raises(ValueError):
        validate_manifest(manifest, partitioned=True)


def test_missing_patient_rejected(manifest):
    manifest.loc[0, "patient_id"] = ""
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_pca_fit_is_exactly_train_only(manifest, tmp_path):
    train = manifest[manifest.partition.eq("train")]
    x = np.random.default_rng(2).normal(size=(len(train), 8))
    pca = TrainingPCA(3).fit(x, train.image_id, manifest)
    np.testing.assert_allclose(pca.mean, x.mean(axis=0))
    before = pca.components.copy()
    pca.transform(np.full((4, 8), 1e9))
    np.testing.assert_array_equal(before, pca.components)
    with pytest.raises(ValueError):
        TrainingPCA(3).fit(np.zeros((len(manifest), 8)), manifest.image_id, manifest)
    with pytest.raises(ValueError):
        TrainingPCA(3).fit(x[:3], train.image_id.iloc[:3], manifest)
    with pytest.raises(ValueError):
        pca.fit(x, train.image_id, manifest)
    pca.save(tmp_path/"pca.npz")
    restored = TrainingPCA.load(tmp_path/"pca.npz")
    np.testing.assert_allclose(restored.transform(x), pca.transform(x))
    assert set(restored.fit_ids) == set(train.image_id)


def test_pca_matches_training_only_sklearn(manifest):
    from sklearn.decomposition import PCA
    frame = manifest[manifest.partition.eq("train")]
    x = np.random.default_rng(5).normal(size=(len(frame), 6))
    guarded = TrainingPCA(2).fit(x, frame.image_id, manifest)
    reference = PCA(2, svd_solver="full").fit(x)
    np.testing.assert_allclose(guarded.transform(x), reference.transform(x), atol=1e-6)


def test_signed_hu_and_display_polarity(dicom):
    expected = [[-1200, -1000], [0, 1000]]
    np.testing.assert_array_equal(dicom_to_hu(dicom), expected)
    dicom.PhotometricInterpretation = "MONOCHROME1"
    np.testing.assert_array_equal(dicom_to_hu(dicom), expected)
    dicom.PixelPaddingValue = -100
    assert dicom_to_hu(dicom)[0, 0] == -1024


def test_invalid_dicom_fails_loudly(dicom):
    del dicom.RescaleSlope
    with pytest.raises(ValueError):
        dicom_to_hu(dicom)


def test_constant_images_finite_and_fixed_windows():
    image = window_image(np.full((2, 2), 40), (2, 2))
    assert image.dtype == np.float32 and np.isfinite(image).all()
    np.testing.assert_allclose(image[0, 0], [127.5, 76.5, 127.5], atol=1e-5)


def test_labels_not_argmax_and_undefined_metrics():
    y = np.array([[1,1,0,0,0,1], [0,0,0,0,0,0]])
    p = np.array([[0.9,0.8,0.1,0.1,0.1,0.9], [0.1]*6])
    report = evaluate_predictions(y, p)
    assert report["exact_match"] == 1 and report["micro_f1"] == 1
    assert report["per_label"]["intraparenchymal"]["tp"] == 1
    assert report["per_label"]["subdural"]["roc_auc"] is None
    assert report["per_label"]["subdural"]["sensitivity"] is None


def test_label_duplicates_and_missing_labels(tmp_path):
    path = tmp_path/"labels.csv"
    path.write_text("ID,Label\nID_a_any,1\nID_a_any,0\n")
    with pytest.raises(ValueError, match="Conflicting"):
        read_labels(path)
    path.write_text("ID,Label\nID_a_any,1\n")
    with pytest.raises(ValueError):
        read_labels(path)


def test_loss_weights_reject_heldout(manifest):
    with pytest.raises(ValueError, match="training"):
        training_positive_weights(manifest)
