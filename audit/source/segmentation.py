# CELL 0
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pydicom
import glob


# Path to your trained classification model 
model_path = '/kaggle/input/ensemblemodel/keras/default/1/ensemble_model.h5'
model = load_model(model_path)
model.summary()
# CELL 1
def load_dicom_as_array(dicom_path, target_size=(224, 224)):
    dicom = pydicom.dcmread(dicom_path)
    img = dicom.pixel_array.astype(np.float32)

    # Normalize image (optional: apply windowing)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # Resize
    img_resized = cv2.resize(img, target_size)
    img_resized = np.expand_dims(img_resized, axis=-1)
    img_rgb = np.repeat(img_resized, 3, axis=-1)  # Make it 3-channel
    img_rgb = np.expand_dims(img_rgb, axis=0)     # Add batch dimension
    return img_rgb, img_resized  # return both preprocessed and original

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # Compute mean intensity
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply channel weights with conv layer output
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_pseudo_mask(heatmap, original_img, output_path, threshold=0.5):
    heatmap_resized = cv2.resize(heatmap, original_img.shape[::-1])
    mask = (heatmap_resized > threshold).astype(np.uint8)

    # Save mask
    cv2.imwrite(output_path, mask * 255)

    # Display
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original_img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.subplot(1, 3, 3)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.show()

dicom_path = "/kaggle/input/images/1-1.dcm"
img, original = load_dicom_as_array(dicom_path)

# Replace this with your last conv layer name (e.g., 'conv5_block3_out' for ResNet50)
last_conv_layer = 'conv5_block3_out'

heatmap = make_gradcam_heatmap(img, model, last_conv_layer)
save_pseudo_mask(heatmap, original, output_path='mask.png')
# CELL 2
dicom_dir = '/kaggle/input/your-dicom-folder/'
output_dir = '/kaggle/working/masks/'

os.makedirs(output_dir, exist_ok=True)
dicom_files = glob.glob(os.path.join(dicom_dir, '*.dcm'))

for i, dicom_path in enumerate(dicom_files):
    try:
        img, original = load_dicom_as_array(dicom_path)
        heatmap = make_gradcam_heatmap(img, model, last_conv_layer)
        output_path = os.path.join(output_dir, f'mask_{i}.png')
        save_pseudo_mask(heatmap, original, output_path)
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")
