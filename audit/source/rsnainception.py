# CELL 0
!pip install scikit-image pydicom opencv-python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence

# Dataset Path
base_path = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"
train_images_path = os.path.join(base_path, 'stage_2_train/')
train_csv = os.path.join(base_path, 'stage_2_train.csv')


# Load and preprocess train.csv
df = pd.read_csv(train_csv)
# Create 'id' and 'subtype'
df['id'] = df['ID'].apply(lambda st: "ID_" + st.split('_')[1])
df['subtype'] = df['ID'].apply(lambda st: st.split('_')[2])
df = df[["id", "subtype", "Label"]]


# Remove duplicates and fill missing values
df = df.drop_duplicates(subset=['id', 'subtype'])
df['Label'] = df['Label'].fillna(0)


# Use only a subset
df_small = df.sample(3000, random_state=42).reset_index(drop=True)
print(f"Using {len(df_small)} samples for training.")

# Image preprocessing function
def load_dicom_image(path, size=(299, 299)):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    img = cv2.resize(img, size)
    img = img.astype(np.float32)
    img -= np.min(img)
    img /= np.max(img)
    img = np.stack([img]*3, axis=-1)  # 3-channel for InceptionV3
    return img


class RSNADataset(Sequence):
    def __init__(self, df, batch_size=32, augment=False):
        self.df = df
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.zeros((len(batch_df), 299, 299, 3), dtype=np.float32)
        y = np.zeros((len(batch_df), 6), dtype=np.float32)

        for i, row in enumerate(batch_df.itertuples()):
            img_path = os.path.join(train_images_path, f"{row.id}.dcm")  # Use 'id' from your df
            X[i] = load_dicom_image(img_path)
            
            # Extract labels by subtype for one-hot
            subtypes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
            y_row = df[(df['id'] == row.id)]
            labels = np.zeros(6)
            for j, subtype in enumerate(subtypes):
                match = y_row[y_row['subtype'] == subtype]
                if not match.empty:
                    labels[j] = match['Label'].values[0]
            y[i] = labels

        return X, y


# Train/Val split
train_df, val_df = train_test_split(df_small, test_size=0.1, random_state=42)

train_gen = RSNADataset(train_df, batch_size=16)
val_gen = RSNADataset(val_df, batch_size=16)

# Build model using InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_gen,
          validation_data=val_gen,
          epochs=3)

# Save model
model.save('inceptionv3.h5')
# CELL 1
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

# Custom F1 score
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

# Compile with named metrics
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=[
                  'accuracy',
                  Precision(name='precision'),
                  Recall(name='recall'),
                  f1_score  # Custom function will appear as 'f1_score'
              ])

# Retrain
history = model.fit(train_gen, validation_data=val_gen, epochs=3)
# CELL 2
def plot_training_history(history):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f'{metric.title()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)
        plt.show()


plot_training_history(history)
# CELL 3
import numpy as np
import pydicom
import os
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("inceptionv3.h5")

# Path to DICOM image
image_path = "/kaggle/input/images/1-1.dcm"

# Hemorrhage subtypes
hemorrhage_types = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

# Preprocessing function
def preprocess_dicom(file_path, target_size=(224, 224)):
    try:
        dicom = pydicom.dcmread(file_path)
        img = dicom.pixel_array
        img = resize(img, target_size, mode="constant", anti_aliasing=True)
        img = np.expand_dims(img, axis=-1)       # shape: (224, 224, 1)
        img = np.repeat(img, 3, axis=-1)         # Convert to RGB: (224, 224, 3)
        img = img / np.max(img) if np.max(img) > 0 else img  # Normalize to [0, 1]
        return np.expand_dims(img, axis=0), dicom.pixel_array
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Predict and display
image, original_image = preprocess_dicom(image_path)
if image is not None:
    # Predict
    predictions = model.predict(image)[0]

    # Display predictions
    print("\nPrediction Results:")
    for hemorrhage, prob in zip(hemorrhage_types, predictions):
        status = "Hemorrhage Detected" if prob > 0.5 else "No Hemorrhage"
        print(f"{hemorrhage}: {status} ({prob:.2f})")

    # Show image
    plt.imshow(original_image, cmap='gray')
    plt.title("DICOM Image")
    plt.axis("off")
    plt.show()
else:
    print("Failed to preprocess the image.")
