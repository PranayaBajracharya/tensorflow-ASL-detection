import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf  # Import TensorFlow
from tensorflow import keras  # For Keras components
from keras import layers  # For layers
from sklearn.model_selection import KFold
import keras_tuner as kt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# Path to dataset (adjust as needed)S
data_dir = '/home/pranaya/data science project/data/asl_alphabet_train/'

# Load images and labels
images = []
labels = []
class_names = sorted(os.listdir(data_dir))  # Assumes folders are class names
for label, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            images.append(img)
            labels.append(label)

images = np.array(images).reshape(-1, 64, 64, 1)  # Add channel dimension
labels = np.array(labels)

# Normalize pixel values to [0,1]
images = images / 255.0

# Split into train/val/test (70:15:15)
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Function for data augmentation (flip, brightness, contrast)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)  # Horizontal flip
    image = tf.image.random_flip_up_down(image)  # Vertical flip
    image = tf.image.random_brightness(image, 0.2)  # Random brightness
    image = tf.image.random_contrast(image, 0.8, 1.2)  # Random contrast
    return image, label

# Create tf.data.Dataset for training with augmentation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1000).map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

# Create tf.data.Dataset for validation (no augmentation)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

# Create tf.data.Dataset for testing (no augmentation)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)