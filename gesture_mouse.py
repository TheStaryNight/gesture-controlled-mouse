import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import pickle
import os
import urllib.request
import threading
from collections import Counter
from pynput.mouse import Controller, Button
from screeninfo import get_monitors

MODEL_FILE = "gesture_model.pkl"
MODEL_PATH = "hand_landmarker.task"

# User-requested gesture mappings
CLASSES = {
    0: "Move (Hover) [Index Up]",
    1: "Left Click [Index + Thumb Pinch]",
    2: "Right Click [Index + Pinky Up]",
    3: "Scroll Mode [Index + Middle Up]"
}

# Verify model file exists
if not os.path.exists(MODEL_FILE):
    print(f"Warning: Trained model '{MODEL_FILE}' not found!")
    print("Please record your custom gestures using collect_data.py, then run train_model.py to generate the model.")

# Check and download model asset if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

# Load the trained machine learning classifier (only if it exists)
clf_model = None
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        clf_model = pickle.load(f)
    print(f"Successfully loaded trained AI model: {type(clf_model).__name__}")

# Initialize mouse controller
mouse = Controller()

# Fetch primary monitor dimensions
monitors = get_monitors()
if not monitors:
    print("Error: No monitors detected!")
    exit(1)
primary_monitor = monitors[0]
SCREEN_WIDTH = primary_monitor.width
SCREEN_HEIGHT = primary_monitor.height
print(f"Screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# High-Performance Threaded Video Stream (eliminates camera I/O lag, boosting FPS to 60)
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

# Initialize MediaPipe Tasks Hand Landmarker (with GPU hardware acceleration support)
try:
    print("Initializing Hand Landmarker with GPU support...")
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH,
        delegate=python.BaseOptions.Delegate.GPU
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
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
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

# Define tracking/touchpad box boundaries (relative to camera dimensions)
# Only moving within this box moves the cursor, allowing high sensitivity/smaller physical movement
BOX_X1, BOX_Y1 = 150, 80
BOX_X2, BOX_Y2 = 490, 360
BOX_WIDTH = BOX_X2 - BOX_X1
BOX_HEIGHT = BOX_Y2 - BOX_Y1

# Coordinate Smoothing (Exponential moving average)
# Lower values make the cursor smoother, but introduce lag.
# E.g., 0.22 is a good compromise.
SMOOTHING = 0.22
prev_x, prev_y = 0, 0

# Debouncing Configuration
# We keep a history of raw model predictions to filter out frame-by-frame class noise
DEBOUNCE_WINDOW = 6
prediction_history = []

# Action State Machine
left_clicked = False
right_clicked = False

# Scroll State
prev_scroll_y = None
SCROLL_THRESHOLD = 0.01
SCROLL_SPEED = 200

# FPS Calculation
prev_time = 0

# Connections map for custom high-tech HUD drawing
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (9, 10), (10, 11), (11, 12),         # Middle
    (13, 14), (14, 15), (15, 16),        # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17)            # Palm base
]

def draw_hand_custom(img_frame, landmarks_list):
    """Draw high-tech glowing skeleton on detected hand landmarks."""
    h, w, c = img_frame.shape
    # Draw connection lines
    for start_idx, end_idx in CONNECTIONS:
        p1 = landmarks_list[start_idx]
        p2 = landmarks_list[end_idx]
        pt1 = (int(p1.x * w), int(p1.y * h))
        pt2 = (int(p2.x * w), int(p2.y * h))
        cv2.line(img_frame, pt1, pt2, (255, 230, 100), 2)  # Glowing cyan-blue
        
    # Draw joint points
    for idx, lm in enumerate(landmarks_list):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in [4, 8, 12, 20]:  # Important controller tips
            cv2.circle(img_frame, (cx, cy), 6, (0, 0, 255), -1)  # Red tips
        else:
            cv2.circle(img_frame, (cx, cy), 4, (0, 255, 0), -1)  # Green joints

# Start threaded video stream
vs = ThreadedVideoStream(0).start()
time.sleep(1.0)  # Let camera initialize

print("Launching Virtual Mouse... Press 'q' in the camera window to quit.")

while True:
    frame = vs.read()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    current_mode_text = "NO HAND DETECTED"
    mode_color = (128, 128, 128)

    # Draw active tracking boundary box on screen
    cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (255, 255, 255), 2)

    if results.hand_landmarks:
        # Get the first detected hand
        landmarks = results.hand_landmarks[0]
        
        # Draw custom skeleton glow
        draw_hand_custom(frame, landmarks)
        
        wrist = landmarks[0]

        # ----------------------------------------------------
        # 1. Feature Extraction & Normalization
        # ----------------------------------------------------
        rel_coords = []
        distances = []
        for lm in landmarks:
            rx = lm.x - wrist.x
            ry = lm.y - wrist.y
            rel_coords.append((rx, ry))
            distances.append(np.sqrt(rx**2 + ry**2))

        # Epsilon-safe scale normalization
        max_dist = max(distances)
        if max_dist < 1e-5:
            max_dist = 1e-5

        normalized_features = []
        for rx, ry in rel_coords:
            normalized_features.append(rx / max_dist)
            normalized_features.append(ry / max_dist)

        # ----------------------------------------------------
        # 2. ML Inference & Debouncing
        # ----------------------------------------------------
        predicted_class = 0  # Default to Hover if model not loaded
        
        if clf_model is not None:
            raw_prediction = int(clf_model.predict([normalized_features])[0])
            
            # Debouncing: majority voting across the last N predictions
            prediction_history.append(raw_prediction)
            if len(prediction_history) > DEBOUNCE_WINDOW:
                prediction_history.pop(0)
                
            # Mode of the window represents the debounced gesture class
            predicted_class = Counter(prediction_history).most_common(1)[0][0]

        # ----------------------------------------------------
        # 3. Mouse Coordinate Tracking & Smoothing
        # ----------------------------------------------------
        # The cursor tracks the index finger tip (landmark 8) coordinates directly
        ix, iy = landmarks[8].x * w, landmarks[8].y * h

        # Map the coordinates relative to our touchpad box boundaries
        ix_clamped = np.clip(ix, BOX_X1, BOX_X2)
        iy_clamped = np.clip(iy, BOX_Y1, BOX_Y2)

        # Convert to [0.0, 1.0] inside the box
        norm_x = (ix_clamped - BOX_X1) / BOX_WIDTH
        norm_y = (iy_clamped - BOX_Y1) / BOX_HEIGHT

        # Interpolate to target screen coordinates
        target_x = norm_x * SCREEN_WIDTH
        target_y = norm_y * SCREEN_HEIGHT

        # Smooth cursor movement to eliminate coordinate noise
        smoothed_x = prev_x + SMOOTHING * (target_x - prev_x)
        smoothed_y = prev_y + SMOOTHING * (target_y - prev_y)
        prev_x, prev_y = smoothed_x, smoothed_y

        # Always move the cursor when hand is detected, unless scrolling
        mouse.position = (int(smoothed_x), int(smoothed_y))

        # ----------------------------------------------------
        # 4. Gesture to Action Mapping State Machine
        # ----------------------------------------------------
        
        # [Class 0: CURSOR MOVE]
        if predicted_class == 0:
            current_mode_text = CLASSES[0]
            mode_color = (0, 255, 0)  # Green
            
            # Cleanup click and scroll states
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False
            if right_clicked:
                right_clicked = False
            prev_scroll_y = None

        # [Class 1: LEFT CLICK & DRAG]
        elif predicted_class == 1:
            current_mode_text = CLASSES[1]
            mode_color = (0, 0, 255)  # Red
            
            if not left_clicked:
                mouse.press(Button.left)
                left_clicked = True
            if right_clicked:
                right_clicked = False
            prev_scroll_y = None

        # [Class 2: RIGHT CLICK]
        elif predicted_class == 2:
            current_mode_text = CLASSES[2]
            mode_color = (0, 255, 255)  # Yellow
            
            if not right_clicked:
                mouse.click(Button.right, 1)
                right_clicked = True
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False
            prev_scroll_y = None

        # [Class 3: SCROLL MODE]
        elif predicted_class == 3:
            current_mode_text = CLASSES[3]
            mode_color = (255, 0, 0)  # Blue
            
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False
            if right_clicked:
                right_clicked = False

            # Handle scrolling logic
            current_scroll_y = landmarks[8].y
            if prev_scroll_y is not None:
                diff = prev_scroll_y - current_scroll_y
                if abs(diff) > SCROLL_THRESHOLD:
                    scroll_amount = int(diff * SCROLL_SPEED)
                    if scroll_amount != 0:
                        mouse.scroll(0, scroll_amount)
            prev_scroll_y = current_scroll_y

        # Highlight tracking coordinate (Index tip) in HUD
        cv2.circle(frame, (int(ix), int(iy)), 12, mode_color, -1)

    else:
        # Reset prediction history and mouse states when no hand is present
        prediction_history.clear()
        if left_clicked:
            mouse.release(Button.left)
            left_clicked = False
        right_clicked = False
        prev_scroll_y = None

    # Calculate and render FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    
    # HUD details
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"MODE: {current_mode_text}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    cv2.putText(frame, "Custom Gestures (ML-Driven):", (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "0 -> Move Cursor [Index Up]", (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "1 -> Left Click [Index + Thumb Pinch]", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "2 -> Right Click [Index + Pinky Up]", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, "3 -> Scroll Mode [Index + Middle Up]", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show video frame
    cv2.imshow("Gesture Control Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
print("Virtual Mouse shut down successfully.")
