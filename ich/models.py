"""Original baseline architectures with explicit, serialized ImageNet adapters."""
import tensorflow as tf

CNN_NAMES = ("vgg16", "resnet50", "densenet121", "efficientnetb0", "inceptionv3")
BASELINES = (*CNN_NAMES, "ensemble", "pca_ensemble")
PCA_BRANCHES = ("resnet50", "efficientnetb0", "inceptionv3")


@tf.keras.utils.register_keras_serializable(package="ich")
class ImageNetInput(tf.keras.layers.Layer):
    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        if backbone not in CNN_NAMES:
            raise ValueError(backbone)
        self.backbone = backbone

    def call(self, inputs):
        app = tf.keras.applications
        functions = {"vgg16": app.vgg16.preprocess_input, "resnet50": app.resnet50.preprocess_input,
                     "densenet121": app.densenet.preprocess_input, "inceptionv3": app.inception_v3.preprocess_input,
                     "efficientnetb0": app.efficientnet.preprocess_input}
        return functions[self.backbone](tf.cast(inputs, tf.float32))

    def get_config(self):
        return dict(super().get_config(), backbone=self.backbone)


def image_size(name):
    return (299, 299) if name == "inceptionv3" else (224, 224)


def feature_model(branches, size=(224, 224), weights="imagenet"):
    inputs = tf.keras.Input((*size, 3), name="hu_windows_0_255")
    app = tf.keras.applications
    factories = dict(zip(CNN_NAMES, (app.VGG16, app.ResNet50, app.DenseNet121, app.EfficientNetB0, app.InceptionV3)))
    features = []
    for name in branches:
        adapted = ImageNetInput(name, name=f"{name}_preprocess")(inputs)
        base = factories[name](include_top=False, weights=weights, input_tensor=adapted)
        base.trainable = False
        features.append(tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_pool")(base.output))
    output = features[0] if len(features) == 1 else tf.keras.layers.Concatenate(name="cnn_features")(features)
    return tf.keras.Model(inputs, output, name="feature_extractor")


def build_model(name, weights="imagenet", size=None):
    if name not in BASELINES or name == "pca_ensemble":
        raise ValueError("Use build_pca_head for PCA ensemble")
    branches = ("efficientnetb0", "resnet50") if name == "ensemble" else (name,)
    extractor = feature_model(branches, size or image_size(name), weights)
    width = 256 if name == "ensemble" else 1024 if name == "inceptionv3" else 128
    x = tf.keras.layers.Dense(width, activation="relu")(extractor.output)
    if name != "inceptionv3":
        x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(6, activation="sigmoid", name="hemorrhage")(x)
    return tf.keras.Model(extractor.input, output, name=name)


def build_pca_head(components):
    return tf.keras.Sequential([
        tf.keras.Input((components,)), tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(6, activation="sigmoid")], name="pca_head")

