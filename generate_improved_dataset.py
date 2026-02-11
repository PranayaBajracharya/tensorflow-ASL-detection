import os
import numpy as np
import cv2
from pathlib import Path

# ===================== IMPROVED DATA GENERATION =====================
print("=" * 70)
print("GENERATING IMPROVED ASL ALPHABET DATASET")
print("=" * 70)

data_dir = '/home/pranaya/data science project/data/asl_alphabet_train/'
os.makedirs(data_dir, exist_ok=True)

# Create folders for each letter A-Z
letters = [chr(65 + i) for i in range(26)]  # A-Z
images_per_letter = 150  # Increased from 10 to 150

print(f"\nGenerating {images_per_letter} images per letter ({len(letters)} letters)")
print(f"Total images: {len(letters) * images_per_letter}\n")

def generate_hand_shape(center, size, rotation=0):
    """Generate a more realistic hand-like shape"""
    mask = np.zeros((64, 64), dtype=np.uint8)
    
    # Main hand body (ellipse)
    hand_center = center
    axes = (size, size // 2)
    cv2.ellipse(mask, hand_center, axes, rotation, 0, 360, 255, -1)
    
    # Add fingers (small circles at the top)
    finger_spacing = size // 4
    finger_y = center[1] - size // 2 - 5
    
    finger_positions = [
        (center[0] - finger_spacing, finger_y),
        (center[0], finger_y - 5),
        (center[0] + finger_spacing, finger_y),
    ]
    
    for pos in finger_positions:
        cv2.circle(mask, pos, size // 5, 255, -1)
    
    return mask

def add_realistic_effects(img, intensity=1.0):
    """Add realistic effects like noise, blur, and brightness variation"""
    # Random noise
    noise = np.random.normal(0, 15 * intensity, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Random blur
    if np.random.rand() > 0.5:
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Random brightness
    brightness = np.random.uniform(0.7, 1.3)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Random contrast
    contrast = np.random.uniform(0.8, 1.2)
    img = np.clip(128 + contrast * (img - 128), 0, 255).astype(np.uint8)
    
    return img

def create_varied_hand_image(letter_idx, image_num):
    """Create varied hand images for each letter"""
    img = np.zeros((64, 64), dtype=np.uint8)
    
    # Background with gradient
    for i in range(64):
        img[i, :] = np.linspace(20 + image_num % 50, 60 + image_num % 40, 64).astype(np.uint8)
    
    # Vary hand position
    center_x = np.random.randint(20, 44)
    center_y = np.random.randint(20, 44)
    center = (center_x, center_y)
    
    # Vary hand size
    size = np.random.randint(10, 20)
    
    # Vary rotation
    rotation = np.random.uniform(0, 360)
    
    # Generate hand shape
    hand_mask = generate_hand_shape(center, size, rotation)
    
    # Blend hand with background based on letter
    hand_intensity = 150 + letter_idx * 3  # Different intensity per letter
    img = cv2.addWeighted(img, 0.4, hand_mask, 0.6, 0)
    img = np.clip(img + (hand_mask * hand_intensity // 255 * 0.5), 0, 255).astype(np.uint8)
    
    # Add some structure lines (to simulate finger details)
    num_lines = np.random.randint(2, 5)
    for _ in range(num_lines):
        x1, y1 = np.random.randint(10, 54, 2)
        x2, y2 = np.random.randint(10, 54, 2)
        cv2.line(img, (x1, y1), (x2, y2), 
                np.random.randint(100, 200), 
                np.random.randint(1, 3))
    
    # Add letter-specific features (subtle variations)
    letter_feature = (letter_idx % 5) * 40
    cv2.circle(img, (32 + (letter_idx % 3) * 5, 32), 3, min(200, letter_feature), -1)
    
    # Add realistic effects
    img = add_realistic_effects(img, intensity=1.0)
    
    # Random rotation (small)
    if np.random.rand() > 0.6:
        angle = np.random.uniform(-15, 15)
        matrix = cv2.getRotationMatrix2D((32, 32), angle, 1)
        img = cv2.warpAffine(img, matrix, (64, 64))
    
    # Random shift
    if np.random.rand() > 0.5:
        shift_x = np.random.randint(-5, 6)
        shift_y = np.random.randint(-5, 6)
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, matrix, (64, 64))
    
    return img

# Generate all images
total_generated = 0
for letter in letters:
    letter_dir = os.path.join(data_dir, letter)
    os.makedirs(letter_dir, exist_ok=True)
    
    letter_idx = ord(letter) - ord('A')
    
    # Clear existing images in this folder (keep only new ones)
    existing_files = os.listdir(letter_dir)
    for f in existing_files:
        os.remove(os.path.join(letter_dir, f))
    
    # Generate new images
    for i in range(images_per_letter):
        img = create_varied_hand_image(letter_idx, i)
        
        # Save image
        img_path = os.path.join(letter_dir, f'{letter}_{i:03d}.jpg')
        cv2.imwrite(img_path, img)
        
        total_generated += 1
        
        # Progress indicator
        if (i + 1) % 30 == 0:
            print(f"  {letter}: {i + 1}/{images_per_letter} images", end="\r")
    
    print(f"✓ Letter '{letter}': {images_per_letter} images generated")

# ===================== DATASET STATISTICS =====================
print(f"\n{'='*70}")
print(f"✅ DATASET GENERATION COMPLETE!")
print(f"{'='*70}")
print(f"\nDataset Statistics:")
print(f"  • Total images: {total_generated}")
print(f"  • Total classes: {len(letters)}")
print(f"  • Images per class: {images_per_letter}")
print(f"  • Image resolution: 64×64 pixels")
print(f"  • Image format: Grayscale JPG")
print(f"  • Dataset location: {data_dir}")

# Verify dataset
print(f"\nDataset Verification:")
total_files = 0
for letter in letters:
    letter_dir = os.path.join(data_dir, letter)
    num_files = len([f for f in os.listdir(letter_dir) if f.endswith('.jpg')])
    total_files += num_files
    status = "✓" if num_files == images_per_letter else "✗"
    print(f"  {status} {letter}: {num_files} images")

print(f"\nTotal verified: {total_files} images from {len(letters)} classes")
print(f"Storage usage: ~{(total_files * 4) / 1024:.1f} MB")

print(f"\n{'='*70}")
print("Next steps:")
print("  1. Run training3.py to train the model with new dataset")
print("  2. Run evaluation4.py to evaluate model performance")
print("  3. Run webcam_detection.py to test with your webcam")
print(f"{'='*70}")
