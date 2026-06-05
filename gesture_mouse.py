import cv2
import mediapipe as mp
import numpy as np
import time
from pynput.mouse import Controller, Button
from screeninfo import get_monitors

# Initialize pynput mouse controller
mouse = Controller()

# Fetch primary monitor dimensions
monitors = get_monitors()
if not monitors:
    print("Error: No monitors detected!")
    exit(1)
primary_monitor = monitors[0]
SCREEN_WIDTH = primary_monitor.width
SCREEN_HEIGHT = primary_monitor.height
print(f"Screen resolution detected: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Camera settings
CAM_WIDTH, CAM_HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Define tracking/touchpad box boundaries in camera coordinates
# A central box ensures less physical hand movement is required to cover the screen
BOX_X1, BOX_Y1 = 150, 80
BOX_X2, BOX_Y2 = 490, 360
BOX_WIDTH = BOX_X2 - BOX_X1
BOX_HEIGHT = BOX_Y2 - BOX_Y1

# Smoothing settings (Exponential moving average)
# Lower value = smoother movement but slight delay
# Higher value = faster response but more jitter
SMOOTHING = 0.25
prev_x, prev_y = 0, 0

# State variables for mouse clicks
left_clicked = False
right_clicked = False

# Scroll state variables
prev_scroll_y = None
SCROLL_THRESHOLD = 0.01
SCROLL_SPEED = 200

# FPS calculation
prev_time = 0

def get_distance(p1, p2):
    """Calculate Euclidean distance between two 3D landmarks."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def is_finger_up(landmarks, finger_name):
    """
    Check if a specific finger is extended (up) or folded (down).
    Uses relative height comparison for index, middle, ring, pinky.
    Uses wrist-to-tip distance for the thumb.
    """
    # Landmarks map:
    # Index: Tip (8), PIP (6)
    # Middle: Tip (12), PIP (10)
    # Ring: Tip (16), PIP (14)
    # Pinky: Tip (20), PIP (18)
    if finger_name == "INDEX":
        return landmarks[8].y < landmarks[6].y
    elif finger_name == "MIDDLE":
        return landmarks[12].y < landmarks[10].y
    elif finger_name == "RING":
        return landmarks[16].y < landmarks[14].y
    elif finger_name == "PINKY":
        return landmarks[20].y < landmarks[18].y
    elif finger_name == "THUMB":
        # Check if thumb tip (4) is extended outward from joint (2) relative to wrist (0)
        dist_wrist_tip = np.sqrt((landmarks[4].x - landmarks[0].x)**2 + (landmarks[4].y - landmarks[0].y)**2)
        dist_wrist_joint = np.sqrt((landmarks[2].x - landmarks[0].x)**2 + (landmarks[2].y - landmarks[0].y)**2)
        return dist_wrist_tip > dist_wrist_joint
    return False

print("Launching Virtual Mouse... Press 'q' in the camera window to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read from webcam.")
        break

    # Flip the frame horizontally to match mirrored natural movements
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert the BGR frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_mode = "NO HAND DETECTED"
    mode_color = (128, 128, 128)

    # Draw tracking/touchpad box
    cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Draw hand skeletal overlay
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect which fingers are extended
            index_up = is_finger_up(landmarks, "INDEX")
            middle_up = is_finger_up(landmarks, "MIDDLE")
            ring_up = is_finger_up(landmarks, "RING")
            pinky_up = is_finger_up(landmarks, "PINKY")
            thumb_up = is_finger_up(landmarks, "THUMB")

            # Reference length for scale-independent distance thresholding (Wrist to Index MCP)
            ref_dist = get_distance(landmarks[0], landmarks[5])

            # ----------------------------------------------------
            # 1. LEFT CLICK & DRAG GESTURE
            # Gesture: Index + Middle finger UP & touching (pinched)
            # ----------------------------------------------------
            if index_up and middle_up and not ring_up and not pinky_up:
                dist_index_middle = get_distance(landmarks[8], landmarks[12])
                ratio = dist_index_middle / ref_dist

                # If fingers are pinched together (ratio < threshold)
                if ratio < 0.28:
                    current_mode = "LEFT CLICK / DRAG"
                    mode_color = (0, 0, 255)  # Red
                    if not left_clicked:
                        mouse.press(Button.left)
                        left_clicked = True
                else:
                    # If extended but not touching -> SCROLL MODE
                    current_mode = "SCROLL MODE"
                    mode_color = (255, 0, 0)  # Blue
                    
                    # Release left click if it was active
                    if left_clicked:
                        mouse.release(Button.left)
                        left_clicked = False
                    
                    # Handle scrolling logic
                    current_scroll_y = landmarks[8].y
                    if prev_scroll_y is not None:
                        diff = prev_scroll_y - current_scroll_y
                        if abs(diff) > SCROLL_THRESHOLD:
                            scroll_amount = int(diff * SCROLL_SPEED)
                            if scroll_amount != 0:
                                mouse.scroll(0, scroll_amount)
                    prev_scroll_y = current_scroll_y

            # ----------------------------------------------------
            # 2. RIGHT CLICK GESTURE
            # Gesture: Thumb + Index finger UP & touching (pinched)
            # ----------------------------------------------------
            elif thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
                dist_thumb_index = get_distance(landmarks[4], landmarks[8])
                ratio = dist_thumb_index / ref_dist

                if ratio < 0.35:
                    current_mode = "RIGHT CLICK"
                    mode_color = (0, 255, 255)  # Yellow
                    if not right_clicked:
                        mouse.click(Button.right, 1)
                        right_clicked = True
                else:
                    if right_clicked:
                        right_clicked = False
                    current_mode = "RIGHT CLICK PENDING"
                    mode_color = (0, 165, 255)  # Orange

            # ----------------------------------------------------
            # 3. CURSOR MOVEMENT MODE
            # Gesture: Only Index finger UP, others folded
            # ----------------------------------------------------
            elif index_up and not middle_up and not ring_up and not pinky_up:
                current_mode = "CURSOR MOVE"
                mode_color = (0, 255, 0)  # Green

                # Clean up other click states
                if left_clicked:
                    mouse.release(Button.left)
                    left_clicked = False
                prev_scroll_y = None

                # Get index finger tip coordinates
                ix, iy = landmarks[8].x * w, landmarks[8].y * h

                # Map coordinates relative to the tracking box
                # Clamp coordinates to the box size
                ix_clamped = np.clip(ix, BOX_X1, BOX_X2)
                iy_clamped = np.clip(iy, BOX_Y1, BOX_Y2)

                # Normalize coordinates within the box (0.0 to 1.0)
                norm_x = (ix_clamped - BOX_X1) / BOX_WIDTH
                norm_y = (iy_clamped - BOX_Y1) / BOX_HEIGHT

                # Interpolate to full screen resolution
                target_x = norm_x * SCREEN_WIDTH
                target_y = norm_y * SCREEN_HEIGHT

                # Apply exponential smoothing filter
                smoothed_x = prev_x + SMOOTHING * (target_x - prev_x)
                smoothed_y = prev_y + SMOOTHING * (target_y - prev_y)

                # Move cursor to smoothed coordinates
                # Cast to integer since screen pixels are discrete
                mouse.position = (int(smoothed_x), int(smoothed_y))

                # Save current position for next frame smoothing
                prev_x, prev_y = smoothed_x, smoothed_y

                # Draw a highlight circle on the index finger tip
                cv2.circle(frame, (int(ix), int(iy)), 12, mode_color, -1)

            # If no recognized gesture is active, clean up all mouse states
            else:
                current_mode = "NEUTRAL (HOVER)"
                mode_color = (200, 200, 200)
                if left_clicked:
                    mouse.release(Button.left)
                    left_clicked = False
                prev_scroll_y = None

    else:
        # No hand detected, reset click states
        if left_clicked:
            mouse.release(Button.left)
            left_clicked = False
        prev_scroll_y = None

    # Calculate and overlay frame rate (FPS)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display gesture status HUD
    cv2.putText(frame, f"MODE: {current_mode}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    # Instruction hints
    cv2.putText(frame, "Gestures:", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "- Index finger UP: Move cursor", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "- Index + Middle touch: Left Click/Drag", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "- Index + Middle apart: Scroll", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show processed frame in OpenCV Window
    cv2.imshow("Gesture Control Virtual Mouse", frame)

    # Press 'q' to release camera resources and exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Virtual Mouse shut down successfully.")
