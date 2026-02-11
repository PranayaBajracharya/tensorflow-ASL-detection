import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime

# ===================== SETUP =====================
print("=" * 70)
print("REAL-TIME ASL SIGN LANGUAGE DETECTION")
print("=" * 70)

# Load the trained model
model_path = '/home/pranaya/data science project/models/asl_cnn_model.h5'
if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    print("Please run training3.py first to train the model.")
    exit(1)

print("\n1. Loading trained model...")
model = tf.keras.models.load_model(model_path)
print(f"   ✓ Model loaded successfully")
print(f"   ✓ Total parameters: {model.count_params():,.0f}")

# Get class names
data_dir = '/home/pranaya/data science project/data/asl_alphabet_train/'
class_names = sorted(os.listdir(data_dir))
print(f"   ✓ Classes: {', '.join(class_names)}")

# ===================== CAMERA SETUP =====================
print("\n2. Initializing webcam...")
cap = cv2.VideoCapture(0)  # 0 is default camera

if not cap.isOpened():
    print("❌ Error: Cannot access webcam!")
    exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print(f"   ✓ Webcam initialized")
print(f"   ✓ Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# ===================== PARAMETERS =====================
DETECTION_REGION = (100, 100, 400, 400)  # (x, y, width, height) - Region for sign detection
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display prediction

print(f"\n3. Detection settings:")
print(f"   ✓ Detection region: {DETECTION_REGION[2]}x{DETECTION_REGION[3]} pixels")
print(f"   ✓ Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")

# ===================== HELPER FUNCTIONS =====================
def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 64x64
    resized = cv2.resize(gray, (64, 64))
    # Normalize to [0, 1]
    normalized = resized / 255.0
    # Add batch and channel dimensions
    input_data = np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
    return input_data

def get_prediction(frame):
    """Get model prediction for a frame"""
    input_data = preprocess_frame(frame)
    predictions = model.predict(input_data, verbose=0)
    confidence = np.max(predictions)
    class_idx = np.argmax(predictions)
    return class_names[class_idx], confidence, predictions[0]

# ===================== MAIN LOOP =====================
print(f"\n4. Starting real-time detection...")
print("\nControls:")
print("  • Press 'c' to capture and save detected sign")
print("  • Press 'q' to quit")
print("\n" + "=" * 70)

# Create folder for captured images
capture_dir = '/home/pranaya/data science project/captured_signs/'
os.makedirs(capture_dir, exist_ok=True)

frame_count = 0
detection_history = []  # Store last N predictions for smoothing

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame")
            break
        
        frame_count += 1
        
        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Extract detection region
        x, y, w, h = DETECTION_REGION
        detection_region_frame = frame[y:y+h, x:x+w]
        
        # Get prediction
        predicted_class, confidence, all_predictions = get_prediction(detection_region_frame)
        
        # Add to history for smoothing (keep last 5 predictions)
        detection_history.append((predicted_class, confidence))
        if len(detection_history) > 5:
            detection_history.pop(0)
        
        # Calculate smoothed prediction (majority voting)
        if len(detection_history) > 0:
            # Get most common prediction in recent history
            from collections import Counter
            recent_classes = [p[0] for p in detection_history]
            most_common_class = Counter(recent_classes).most_common(1)[0][0]
            avg_confidence = np.mean([p[1] for p in detection_history])
        else:
            most_common_class = predicted_class
            avg_confidence = confidence
        
        # Print predictions to console every 10 frames
        if frame_count % 10 == 0:
            print(f"\rFrame {frame_count} | Prediction: {predicted_class} ({confidence*100:.1f}%) | Smoothed: {most_common_class} ({avg_confidence*100:.1f}%)", end="")
        
        # Draw detection region rectangle
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw prediction text - ALWAYS ON SCREEN
        # Current prediction
        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_color = (0, 255, 0)  # Green for confident
            text = f"{predicted_class} ({confidence*100:.1f}%)"
        else:
            prediction_color = (0, 165, 255)  # Orange for low confidence
            text = f"{predicted_class} ({confidence*100:.1f}%) - Low Confidence"
        
        # Draw large prediction on top
        cv2.rectangle(display_frame, (x, y-60), (x+w, y-10), prediction_color, -1)
        cv2.putText(display_frame, text, (x+10, y-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Draw confidence bar
        confidence_bar_width = int((w - 20) * confidence)
        cv2.rectangle(display_frame, (x+10, y+h+15), (x+10+confidence_bar_width, y+h+35), 
                     prediction_color, -1)
        cv2.rectangle(display_frame, (x+10, y+h+15), (x+w-10, y+h+35), prediction_color, 2)
        cv2.putText(display_frame, f"Confidence: {confidence*100:.1f}%", (x+10, y+h+55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, prediction_color, 1)
        
        # Smoothed prediction
        smooth_text = f"Smoothed: {most_common_class} ({avg_confidence*100:.1f}%)"
        cv2.putText(display_frame, smooth_text, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw top 3 predictions on the right side
        cv2.putText(display_frame, "Top 3 Predictions:", (430, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        top_3_indices = np.argsort(all_predictions)[-3:][::-1]
        for i, idx in enumerate(top_3_indices):
            pred_text = f"{i+1}. {class_names[idx]}: {all_predictions[idx]*100:.1f}%"
            # Highlight the top prediction
            if i == 0:
                cv2.putText(display_frame, pred_text, (430, 70 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, pred_text, (430, 70 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw frame counter and FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(display_frame, f"Frame: {frame_count} | FPS: {fps:.0f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Draw instructions
        cv2.putText(display_frame, "Press 'c' to capture | 'q' to quit | 's' for screenshot", (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Display the frame
        cv2.imshow('ASL Sign Language Detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            print("\n✓ Exiting...")
            break
        
        elif key == ord('c'):  # Capture
            # Save the detection region
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{most_common_class}_{timestamp}.jpg"
            filepath = os.path.join(capture_dir, filename)
            
            # Save original region
            cv2.imwrite(filepath, detection_region_frame)
            
            # Save display frame (with annotations)
            display_filename = f"{most_common_class}_{timestamp}_annotated.jpg"
            display_filepath = os.path.join(capture_dir, display_filename)
            cv2.imwrite(display_filepath, display_frame)
            
            print(f"\n✓ CAPTURED: '{most_common_class}' (Confidence: {avg_confidence*100:.1f}%)")
            print(f"  Saved: {filename}")
        
        elif key == ord('s'):  # Screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            filepath = os.path.join(capture_dir, filename)
            cv2.imwrite(filepath, display_frame)
            print(f"\n✓ Screenshot saved: {filename}")

except KeyboardInterrupt:
    print("\n✓ Interrupted by user")

finally:
    # ===================== CLEANUP =====================
    print("\n5. Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print(f"   ✓ Webcam released")
    print(f"   ✓ Total frames processed: {frame_count}")
    print(f"   ✓ Captured signs saved to: {capture_dir}")
    print("\n" + "=" * 70)
    print("✅ DETECTION COMPLETED")
    print("=" * 70)
