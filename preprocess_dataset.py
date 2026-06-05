import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import glob
import urllib.request

# Paths
DATASET_DIR = os.path.join("dataset", "leapGestRecog")
CSV_FILE = "gesture_data.csv"
MODEL_PATH = "hand_landmarker.task"

# Check and download model if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded successfully.")

# Gesture folder mappings to our target classes:
# Class 0: Move (Hover) -> 06_index (Index finger extended)
# Class 1: Left Click -> 03_fist (Fist closed)
# Class 2: Right Click -> 02_l (L-shape / Index + Thumb extended)
# Class 3: Scroll Mode -> 01_palm (Open palm)
GESTURE_MAPPING = {
    "06_index": 0,
    "03_fist": 1,
    "02_l": 2,
    "01_palm": 3
}

# Initialize MediaPipe Tasks Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_landmarks(landmarks):
    """Normalize hand landmarks to be scale and translation invariant."""
    wrist = landmarks[0]
    
    # 1. Translation: Subtract wrist coordinate to center hand at (0, 0)
    rel_coords = []
    distances = []
    for lm in landmarks:
        rx = lm.x - wrist.x
        ry = lm.y - wrist.y
        rel_coords.append((rx, ry))
        distances.append(np.sqrt(rx**2 + ry**2))
        
    # 2. Scale: Divide by max distance with epsilon to avoid division-by-zero
    max_dist = max(distances)
    if max_dist < 1e-5:
        max_dist = 1e-5
        
    # Build flat 42-dimensional feature vector
    features = []
    for rx, ry in rel_coords:
        features.append(rx / max_dist)
        features.append(ry / max_dist)
        
    return features

print("Starting LeapGestRecog preprocessing...")
print(f"Dataset directory: {DATASET_DIR}")

# Create or overwrite CSV file and write header
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    # 42 feature columns (x0, y0, x1, y1...) + 1 label column
    header = [f"lm_{i}_{coord}" for i in range(21) for coord in ("x", "y")] + ["label"]
    writer.writerow(header)

total_processed = 0
total_success = 0
total_failed = 0

# Subject folders are '00', '01', ..., '09'
subjects = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.isdigit()]
subjects.sort()

print(f"Found {len(subjects)} subject folders: {subjects}")

for subject in subjects:
    subject_path = os.path.join(DATASET_DIR, subject)
    print(f"\nProcessing Subject {subject}...")
    
    for gesture_folder, label_id in GESTURE_MAPPING.items():
        gesture_path = os.path.join(subject_path, gesture_folder)
        if not os.path.exists(gesture_path):
            print(f"  Warning: Path {gesture_path} not found. Skipping.")
            continue
            
        # Get all PNG images in this folder
        images = glob.glob(os.path.join(gesture_path, "*.png"))
        print(f"  Gesture '{gesture_folder}' -> Class {label_id}: Found {len(images)} images.")
        
        for img_path in images:
            total_processed += 1
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                total_failed += 1
                continue
                
            # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            
            # Process image using MediaPipe Tasks
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                # Get the first detected hand
                hand_landmarks = results.hand_landmarks[0]
                features = normalize_landmarks(hand_landmarks)
                
                # Write features and label to CSV
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(features + [label_id])
                    
                total_success += 1
            else:
                total_failed += 1
                
            # Print periodic progress
            if total_processed % 500 == 0:
                print(f"    Progress: Processed {total_processed} images. Success: {total_success}, Failed (Skipped): {total_failed}")

print("\n--- Preprocessing Complete ---")
print(f"Total Images Scanned: {total_processed}")
print(f"Successfully Extracted & Normalized: {total_success} ({(total_success/total_processed)*100:.2f}%)")
print(f"Failed/Skipped (No hand detected): {total_failed} ({(total_failed/total_processed)*100:.2f}%)")
print(f"Dataset saved to '{CSV_FILE}'")
