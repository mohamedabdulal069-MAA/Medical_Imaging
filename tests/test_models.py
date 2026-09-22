import gc
import numpy as np
import pytest
import tensorflow as tf
from ich.models import ImageNetInput, build_model, build_pca_head, feature_model, PCA_BRANCHES
from ich.explainability import gradcam


@pytest.mark.parametrize("name,module", [("vgg16", "vgg16"), ("resnet50", "resnet50"), ("densenet121", "densenet"), ("efficientnetb0", "efficientnet"), ("inceptionv3", "inception_v3")])
def test_preprocessing_matches_keras(name, module, tmp_path):
    x = np.array([[[[0., 100., 255.]]]], np.float32)
    expected = getattr(tf.keras.applications, module).preprocess_input(x.copy())
    np.testing.assert_allclose(ImageNetInput(name)(x).numpy(), expected)
    inputs = tf.keras.Input((1, 1, 3))
    model = tf.keras.Model(inputs, ImageNetInput(name)(inputs))
    model.save(tmp_path/"adapter.keras")
    restored = tf.keras.models.load_model(tmp_path/"adapter.keras", safe_mode=True)
    np.testing.assert_allclose(restored(x).numpy(), expected)


@pytest.mark.parametrize("name", ["vgg16", "resnet50", "densenet121", "efficientnetb0", "inceptionv3", "ensemble"])
def test_all_preserved_image_architectures(name, tmp_path):
    tf.keras.backend.clear_session()
    model = build_model(name, weights=None, size=(75, 75))
    result = model(np.zeros((1, 75, 75, 3), np.float32), training=False).numpy()
    assert result.shape == (1, 6) and np.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    assert len(model.trainable_weights) == 4  # two dense heads; all backbone weights frozen
    if name in ("inceptionv3", "ensemble"):
        model.save(tmp_path/"real-model.keras")
        restored = tf.keras.models.load_model(tmp_path/"real-model.keras", compile=False, safe_mode=True)
        np.testing.assert_allclose(restored(np.zeros((1,75,75,3), np.float32), training=False).numpy(), result, atol=1e-6)
        del restored
    del model
    tf.keras.backend.clear_session()
    gc.collect()


def test_pca_feature_dimensions_and_head():
    model = feature_model(PCA_BRANCHES, size=(75, 75), weights=None)
    assert model.output_shape == (None, 5376)
    head = build_pca_head(4)
    assert head(np.ones((2, 4))).shape == (2, 6)
    tf.keras.backend.clear_session()


def test_gradcam_explicit_label_and_zero_heatmap():
    inputs = tf.keras.Input((8, 8, 3))
    x = tf.keras.layers.Conv2D(2, 3, name="conv", kernel_initializer="zeros")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(6, activation="sigmoid")(x))
    heatmap = gradcam(model, np.zeros((1, 8, 8, 3), np.float32), "conv", 1)
    assert np.isfinite(heatmap).all() and not heatmap.any()
