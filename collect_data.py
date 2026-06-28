import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import threading
import time
import urllib.request

# Define custom gesture classes requested by the user
CLASSES = {
    0: "Move (Hover) [Index Up]",
    1: "Left Click [Index + Thumb Up]",
    2: "Right Click [Index + Pinky Up]",
    3: "Scroll Mode [Index + Middle Up]"
}

CSV_FILE = "gesture_data.csv"
MODEL_PATH = "hand_landmarker.task"

# Ensure model asset exists
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

# High-Performance Threaded Video Stream to eliminate camera frame acquisition lag
class ThreadedVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 60)  # Attempt to set 60 FPS
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (grabbed, frame) = self.stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# Initialize MediaPipe Tasks Hand Landmarker (with GPU acceleration support)
try:
    print("Initializing Hand Landmarker with GPU support...")
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=python.BaseOptions.Delegate.GPU
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("GPU acceleration enabled.")
except Exception as e:
    print(f"GPU initialization failed ({e}). Falling back to CPU...")
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=python.BaseOptions.Delegate.CPU
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)

# Load existing counts from CSV if it exists
counts = {0: 0, 1: 0, 2: 0, 3: 0}
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        for row in reader:
            if row:
                try:
                    label = int(row[-1])
                    if label in counts:
                        counts[label] += 1
                except ValueError:
                    pass

# Start threaded video stream
vs = ThreadedVideoStream(0).start()
time.sleep(1.0)  # Let camera warm up

print("--- Gesture Data Collector ---")
print("Instructions:")
print("1. Strike your custom hand gesture in front of the camera.")
print("2. Hold down the corresponding key to record frames:")
for k, v in CLASSES.items():
    print(f"   Press '{k}' -> Record: {v}")
print("3. Move/rotate your hand slightly while recording to capture variations.")
print("4. Press 'q' to quit.")
print("------------------------------")

while True:
    frame = vs.read()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB and process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    hand_detected = False
    normalized_features = []

    if results.hand_landmarks:
        hand_detected = True
        landmarks = results.hand_landmarks[0]
        wrist = landmarks[0]
        
        # Draw joints
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        
        # Translate & Normalize
        rel_coords = []
        distances = []
        for lm in landmarks:
            rx = lm.x - wrist.x
            ry = lm.y - wrist.y
            rel_coords.append((rx, ry))
            distances.append(np.sqrt(rx**2 + ry**2))
            
        # Epsilon safe-guard
        max_dist = max(distances)
        if max_dist < 1e-5:
            max_dist = 1e-5
            
        for rx, ry in rel_coords:
            normalized_features.append(rx / max_dist)
            normalized_features.append(ry / max_dist)

    key = cv2.waitKey(1) & 0xFF
    recorded_class = None

    if key == ord('q'):
        break
    elif key in [ord('0'), ord('1'), ord('2'), ord('3')] and hand_detected:
        label = int(chr(key))
        recorded_class = label
        
        # Write to CSV
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(normalized_features + [label])
            
        counts[label] += 1

    # Render HUD
    y_offset = 40
    cv2.putText(frame, "Data Collection Counts:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for k, v in CLASSES.items():
        y_offset += 25
        color = (0, 255, 0) if recorded_class == k else (200, 200, 200)
        cv2.putText(frame, f"[{k}] {v}: {counts[k]} frames", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if recorded_class is not None:
        cv2.putText(frame, f"SAVING CLASS {recorded_class}...", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif not hand_detected:
        cv2.putText(frame, "No hand detected - hold hand up to record", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Hold 0, 1, 2, or 3 to record data", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Gesture Data Collector", frame)

vs.stop()
cv2.destroyAllWindows()
print("\nData collection finished.")
for k, v in CLASSES.items():
    print(f"  {v}: {counts[k]} frames")
