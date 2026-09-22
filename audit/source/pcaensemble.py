# CELL 0
!pip install scikit-image pydicom opencv-python

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
import pydicom
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils import class_weight

from sklearn.decomposition import PCA
from tensorflow.keras.layers import Lambda

from sklearn.metrics import f1_score

import joblib 

# Dataset paths
dataset_path = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"
train_images_dir = os.path.join(dataset_path, "stage_2_train/")
train_csv = os.path.join(dataset_path, "stage_2_train.csv")

# Load and preprocess train.csv
df = pd.read_csv(train_csv)
# Create 'id' and 'subtype'
df['id'] = df['ID'].apply(lambda st: "ID_" + st.split('_')[1])
df['subtype'] = df['ID'].apply(lambda st: st.split('_')[2])
df = df[["id", "subtype", "Label"]]


# Remove duplicates and fill missing values
df = df.drop_duplicates(subset=['id', 'subtype'])
df['Label'] = df['Label'].fillna(0)

# Pivot the data
df_pivot = df.pivot(index='id', columns='subtype', values='Label').reset_index()
print("Pivoted DataFrame shape:", df_pivot.shape)
print(df_pivot.head())

# Add file paths
df_pivot['file_path'] = df_pivot['id'].apply(lambda x: os.path.join(train_images_dir, f"{x}.dcm"))

# subset
df_subset = df_pivot.sample(n=3000, random_state=42)
print(f"Using {len(df_subset)} samples for training.")

# Preprocess DICOM images function
def preprocess_dicom(file_path, label, target_size=(224, 224)):
    try:
        dicom = pydicom.dcmread(file_path)
        img = dicom.pixel_array
        img = resize(img, target_size, mode="constant", anti_aliasing=True)
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
        img = img / np.max(img) if np.max(img) > 0 else img
        return img, label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


images, labels = [], []
for _, row in df_subset.iterrows():
    img, lbl = preprocess_dicom(row['file_path'],
                                row[["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]].values)
    if img is not None:
        images.append(img)
        labels.append(lbl)

images = np.array(images)
labels = np.array(labels)
print("Preprocessed Images shape:", images.shape)
print("Preprocessed Labels shape:", labels.shape)

# Split data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)



def create_ensemble_model_with_pca(images, labels, input_shape=(224, 224, 3), num_classes=6, pca_components=512):
    # Step 1: Extract features from both models
    resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=input_shape)
    effnet_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg', input_shape=input_shape)
    inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape=input_shape)
    
    print("Extracting ResNet features...")
    resnet_features = resnet_model.predict(images, batch_size=32, verbose=1)
    
    print("Extracting EfficientNet features...")
    effnet_features = effnet_model.predict(images, batch_size=32, verbose=1)

    print("Extracting InceptionV3 features...")
    inception_features = inception_model.predict(images, batch_size=32, verbose=1)

    # Step 2: Concatenate features
    features = np.concatenate([resnet_features, effnet_features, inception_features], axis=1)
    print("Combined features shape:", features.shape)  # e.g., (5000, 3328)

    # Step 3: Apply PCA
    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features)
    print("Reduced features shape:", features_pca.shape)  # (5000, 128)

    joblib.dump(pca, 'pca_model.pkl')

    # Step 4: Ensure labels are float32
    labels = np.array(labels, dtype=np.float32)

    # Step 5: Split the data
    X_train_pca, X_val_pca, y_train, y_val = train_test_split(features_pca, labels, test_size=0.2, random_state=42)

    # Step 6: Define the classifier model
    model = Sequential([
        tf.keras.layers.Input(shape=(pca_components,)),
        tf.keras.layers.Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Flatten the labels and compute weights for each class
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels.ravel()), y=labels.ravel())
    class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

    # Step 7: Train the model
    model.fit(X_train_pca, y_train, validation_data=(X_val_pca, y_val), epochs=5, batch_size=32, class_weight=class_weights_dict)

    return model

# Call the function and pass `images` and `labels` explicitly
ensemble_model = create_ensemble_model_with_pca(images, labels)

# Save the trained model
ensemble_model.save('ensemble_model.h5')
# CELL 1
import numpy as np
import pydicom
import os
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3

# Load trained models
model = tf.keras.models.load_model("ensemble_model.h5")
pca = joblib.load("pca_model.pkl")


# Define image path
image_path = "/kaggle/input/images/1-1.dcm"

# Define preprocessing
def preprocess_dicom(file_path, target_size=(224, 224)):
    try:
        dicom = pydicom.dcmread(file_path)
        img = dicom.pixel_array
        img = resize(img, target_size, mode="constant", anti_aliasing=True)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.repeat(img, 3, axis=-1)    # Convert grayscale to RGB
        img = img / np.max(img) if np.max(img) > 0 else img  # Normalize
        return np.expand_dims(img, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Define feature extractor
def extract_features(image):
    resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))
    effnet_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))
    inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))

    resnet_feat = resnet_model.predict(image, verbose=0)
    effnet_feat = effnet_model.predict(image, verbose=0)
    inception_feat = inception_model.predict(image, verbose=0)

    features = np.concatenate([resnet_feat, effnet_feat, inception_feat], axis=1)
    return features

# Process the image
image = preprocess_dicom(image_path)
if image is not None:
    # Extract and reduce features
    features = extract_features(image)
    features_pca = pca.transform(features)

    # Predict
    predictions = model.predict(features_pca)[0]

    # Define hemorrhage subtypes
    hemorrhage_types = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

    # Show predictions
    for hemorrhage, prob in zip(hemorrhage_types, predictions):
        print(f"{hemorrhage}: {'Hemorrhage Detected' if prob > 0.5 else 'No Hemorrhage'} ({prob:.2f})")

    # Show image
    dicom = pydicom.dcmread(image_path)
    plt.imshow(dicom.pixel_array, cmap='gray')
    plt.title("DICOM Image")
    plt.axis("off")
    plt.show()
else:
    print("Failed to process the image.")
# CELL 2
from IPython.display import FileLink

# Save model to output
model.save("/kaggle/working/ensemble_model.h5")
FileLink("/kaggle/working/ensemble_model.h5")
