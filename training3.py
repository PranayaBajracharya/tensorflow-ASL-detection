import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers
import matplotlib.pyplot as plt
from datetime import datetime

# Import the CNN model builder
from CNN_model2 import build_cnn_model

# ===================== DATA LOADING =====================
print("=" * 70)
print("TRAINING ASL ALPHABET CNN MODEL")
print("=" * 70)

# Path to dataset
data_dir = '/home/pranaya/data science project/data/asl_alphabet_train/'

# To avoid OOM on very large real datasets, limit images per class during
# early experiments. Set to None to use all images.
MAX_IMAGES_PER_CLASS = 500

print("\n1. Loading dataset...")
# Load images and labels
images = []
labels = []
class_names = sorted(os.listdir(data_dir))  # Assumes folders are class names
for label, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir, class_name)
    count = 0
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            images.append(img)
            labels.append(label)
            count += 1
            if MAX_IMAGES_PER_CLASS is not None and count >= MAX_IMAGES_PER_CLASS:
                break

images = np.array(images).reshape(-1, 64, 64, 1)  # Add channel dimension
labels = np.array(labels)

print(f"   ✓ Loaded {len(images)} images from {len(class_names)} classes")
print(f"   ✓ Image shape: {images.shape}")

# Normalize pixel values to [0,1]
images = images / 255.0
print(f"   ✓ Images normalized to [0, 1]")

# Split into train/val/test (70:15:15)
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\n2. Data split completed:")
print(f"   ✓ Training: {len(X_train)} samples (70%)")
print(f"   ✓ Validation: {len(X_val)} samples (15%)")
print(f"   ✓ Test: {len(X_test)} samples (15%)")

# ===================== DATA AUGMENTATION =====================
print(f"\n3. Setting up data augmentation...")
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

print(f"   ✓ Train dataset: batches of 32 with augmentation")
print(f"   ✓ Val dataset: batches of 32")
print(f"   ✓ Test dataset: batches of 32")

# ===================== MODEL BUILDING =====================
print(f"\n4. Building CNN model...")
model = build_cnn_model(num_classes=len(class_names))
print(f"   ✓ Model created with {model.count_params():,.0f} parameters")

# ===================== CALLBACKS =====================
print(f"\n5. Setting up training callbacks...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]
print(f"   ✓ Early Stopping: patience=10")
print(f"   ✓ Learning Rate Reduction: factor=0.5, patience=5")

# ===================== MODEL TRAINING =====================
print(f"\n6. Training the model...")
print(f"   Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 70)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

print("-" * 70)
print(f"   Complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===================== SAVE MODEL =====================
print(f"\n7. Saving the model...")
model_dir = '/home/pranaya/data science project/models/'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'asl_cnn_model.h5')
model.save(model_path)
print(f"   ✓ Model saved to: {model_path}")

# ===================== TEST SET EVALUATION =====================
print(f"\n8. Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
print(f"   ✓ Test Loss: {test_loss:.4f}")
print(f"   ✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ===================== PLOT TRAINING HISTORY =====================
print(f"\n9. Plotting training history...")
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(model_dir, 'training_history.png')
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
print(f"   ✓ Plot saved to: {plot_path}")

# ===================== SAVE METADATA =====================
print(f"\n10. Saving metadata...")
metadata_path = os.path.join(model_dir, 'model_info.txt')
with open(metadata_path, 'w') as f:
    f.write("ASL Alphabet CNN Model Information\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Number of Classes: {len(class_names)}\n")
    f.write(f"Classes: {', '.join(class_names)}\n")
    f.write(f"Input Shape: (64, 64, 1)\n")
    f.write(f"Total Parameters: {model.count_params():,.0f}\n\n")
    f.write(f"Train Samples: {len(X_train)}\n")
    f.write(f"Val Samples: {len(X_val)}\n")
    f.write(f"Test Samples: {len(X_test)}\n\n")
    f.write(f"Final Train Accuracy: {history.history['accuracy'][-1]:.4f}\n")
    f.write(f"Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")

print(f"   ✓ Metadata saved to: {metadata_path}")

# ===================== SUMMARY =====================
print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nModel saved: {model_path}")
print(f"Training history plot: {plot_path}")
print(f"Model info: {metadata_path}")
print(f"\nNext step: Run evaluation4.py to generate detailed reports")
print("=" * 70)