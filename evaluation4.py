import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== DATA LOADING =====================
print("=" * 70)
print("EVALUATING ASL ALPHABET CNN MODEL")
print("=" * 70)

# Path to dataset
data_dir = '/home/pranaya/data science project/data/asl_alphabet_train/'

print("\n1. Loading dataset...")
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
print(f"   ✓ Loaded {len(images)} images from {len(class_names)} classes")

# Split into train/val/test (70:15:15)
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"   ✓ Test set: {len(X_test)} samples")

# ===================== LOAD MODEL =====================
print(f"\n2. Loading trained model...")
model_path = '/home/pranaya/data science project/models/asl_cnn_model.h5'

if not os.path.exists(model_path):
    print(f"   ✗ Model not found at: {model_path}")
    print(f"   Please run training3.py first to train the model.")
    exit(1)

model = tf.keras.models.load_model(model_path)
print(f"   ✓ Model loaded successfully")
print(f"   ✓ Total parameters: {model.count_params():,.0f}")

# ===================== PREDICTIONS =====================
print(f"\n3. Making predictions on test set...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print(f"   ✓ Predictions completed")

# ===================== ACCURACY METRICS =====================
print(f"\n4. Computing accuracy metrics...")
accuracy = accuracy_score(y_test, y_pred)
print(f"   ✓ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ===================== CONFUSION MATRIX =====================
print(f"\n5. Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - ASL Alphabet Classification', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plot_dir = '/home/pranaya/data science project/models/'
os.makedirs(plot_dir, exist_ok=True)
cm_plot_path = os.path.join(plot_dir, 'confusion_matrix.png')
plt.savefig(cm_plot_path, dpi=100, bbox_inches='tight')
print(f"   ✓ Confusion matrix saved to: {cm_plot_path}")

# ===================== CLASSIFICATION REPORT =====================
print(f"\n6. Generating classification report...")
# Get all unique classes from predictions and labels
all_classes_in_test = np.unique(np.concatenate([y_test, y_pred]))
class_report = classification_report(y_test, y_pred, labels=sorted(all_classes_in_test), 
                                     target_names=[class_names[i] for i in sorted(all_classes_in_test)], 
                                     digits=4, zero_division=0)
print("\nDetailed Classification Report:")
print(class_report)

# Save classification report
report_path = os.path.join(plot_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("ASL Alphabet - Classification Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
    f.write(class_report)
    
    # Add per-class accuracy
    f.write("\n\nPer-Class Accuracy:\n")
    f.write("-" * 70 + "\n")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_accuracy = (y_pred[class_mask] == i).sum() / class_mask.sum()
            f.write(f"{class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)\n")

print(f"   ✓ Classification report saved to: {report_path}")

# ===================== PER-CLASS PERFORMANCE =====================
print(f"\n7. Per-class performance analysis...")
test_classes_indices = [i for i in range(len(class_names)) if (y_test == i).sum() > 0]
test_classes_names = [class_names[i] for i in test_classes_indices]
test_accuracies = []

for i in test_classes_indices:
    class_mask = y_test == i
    if class_mask.sum() > 0:
        class_accuracy = (y_pred[class_mask] == i).sum() / class_mask.sum()
        test_accuracies.append(class_accuracy)

# Plot per-class accuracy (only for classes in test set)
plt.figure(figsize=(16, 6))
colors = ['green' if acc >= 0.9 else 'orange' if acc >= 0.7 else 'red' for acc in test_accuracies]
plt.bar(test_classes_names, test_accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Per-Class Accuracy (Classes in Test Set)', fontsize=16, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim([0, 1.05])
plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

accuracy_plot_path = os.path.join(plot_dir, 'per_class_accuracy.png')
plt.savefig(accuracy_plot_path, dpi=100, bbox_inches='tight')
print(f"   ✓ Per-class accuracy plot saved to: {accuracy_plot_path}")

# ===================== TOP MISCLASSIFICATIONS =====================
print(f"\n8. Analyzing misclassifications...")
misclassified_indices = np.where(y_pred != y_test)[0]
print(f"   ✓ Total misclassifications: {len(misclassified_indices)} / {len(y_test)}")
print(f"   ✓ Misclassification rate: {len(misclassified_indices)/len(y_test)*100:.2f}%")

if len(misclassified_indices) > 0:
    # Get confidence of misclassified predictions
    misclass_confidences = np.max(y_pred_probs[misclassified_indices], axis=1)
    mean_confidence = misclass_confidences.mean()
    print(f"   ✓ Mean confidence on misclassified samples: {mean_confidence:.4f}")

# ===================== SUMMARY REPORT =====================
print(f"\n9. Generating summary report...")
summary_path = os.path.join(plot_dir, 'evaluation_summary.txt')

# Prepare per-class accuracy info
per_class_dict = {}
for i in range(len(class_names)):
    class_mask = y_test == i
    if class_mask.sum() > 0:
        class_accuracy = (y_pred[class_mask] == i).sum() / class_mask.sum()
        per_class_dict[class_names[i]] = class_accuracy

with open(summary_path, 'w') as f:
    f.write("ASL ALPHABET CNN MODEL - EVALUATION SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Total Test Samples: {len(y_test)}\n")
    f.write(f"Correct Predictions: {(y_pred == y_test).sum()}\n")
    f.write(f"Incorrect Predictions: {len(misclassified_indices)}\n")
    f.write(f"Misclassification Rate: {len(misclassified_indices)/len(y_test)*100:.2f}%\n\n")
    
    f.write("BEST PERFORMING CLASSES\n")
    f.write("-" * 70 + "\n")
    if per_class_dict:
        best_classes = sorted(per_class_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (class_name, acc) in enumerate(best_classes, 1):
            f.write(f"{i}. {class_name}: {acc:.4f} ({acc*100:.2f}%)\n")
    else:
        f.write("No classes in test set\n")
    
    f.write("\nCHALLENGING CLASSES\n")
    f.write("-" * 70 + "\n")
    if per_class_dict:
        worst_classes = sorted(per_class_dict.items(), key=lambda x: x[1])[:5]
        for i, (class_name, acc) in enumerate(worst_classes, 1):
            f.write(f"{i}. {class_name}: {acc:.4f} ({acc*100:.2f}%)\n")
    else:
        f.write("No classes in test set\n")
    
    f.write("\nMODEL ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Parameters: {model.count_params():,.0f}\n")
    f.write(f"Input Shape: (64, 64, 1)\n")
    f.write(f"Number of Classes: {len(class_names)}\n")

print(f"   ✓ Summary report saved to: {summary_path}")

# ===================== RESULTS DISPLAY =====================
print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nGenerated Files:")
print(f"  1. Confusion Matrix: {cm_plot_path}")
print(f"  2. Per-Class Accuracy Plot: {accuracy_plot_path}")
print(f"  3. Classification Report: {report_path}")
print(f"  4. Evaluation Summary: {summary_path}")
print(f"\nKey Results:")
print(f"  • Overall Accuracy: {accuracy*100:.2f}%")
print(f"  • Test Samples: {len(y_test)}")
print(f"  • Correct Predictions: {(y_pred == y_test).sum()}")
print(f"  • Misclassifications: {len(misclassified_indices)}")
print("=" * 70)
