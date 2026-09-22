"""Class-specific Grad-CAM for trained image models; not segmentation masks."""
def gradcam(model, images, layer_name, label_index):
    import tensorflow as tf
    if model.output_shape[-1] != 6 or not 0 <= label_index < 6:
        raise ValueError("Select an explicit hemorrhage label from a six-output image model")
    if len(images) != 1:
        raise ValueError("Explain one image at a time")
    layer = model.get_layer(layer_name)
    if len(layer.output.shape) != 4:
        raise ValueError("Choose a spatial convolutional activation")
    probe = tf.keras.Model(model.input, [layer.output, model.output])
    with tf.GradientTape() as tape:
        activations, predictions = probe(images, training=False)
        score = predictions[:, label_index]
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError("Disconnected layer; PCA heads require a differentiable composite model")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))[0]
    return tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap)).numpy()
