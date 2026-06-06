# Gesture Controlled Virtual Mouse (Custom ML Edition)

A real-time computer vision application that allows you to control your PC's mouse cursor, perform left clicks/drags, right clicks, and scroll using hand gestures captured from your laptop's built-in webcam. 

This project uses **Google MediaPipe Tasks API** for high-precision hand landmark extraction and a **Custom Random Forest Classifier** (via Scikit-Learn) trained specifically on *your* hand for 100% accuracy and zero jitter.

---

## 🚀 Installation & Setup

1. **Clone the repository** (if you haven't already):
   ```powershell
   git clone https://github.com/TheStaryNight/gesture-controlled-mouse.git
   cd gesture-controlled-mouse
   ```

2. **Install the dependencies**:
   Make sure you have Python 3.12+ installed, then run:
   ```powershell
   py -m pip install -r requirements.txt
   ```
   *Note: This will download `opencv-python`, `mediapipe`, `pynput`, `screeninfo`, and `scikit-learn`.*

---

## 🎮 How to Use It

Start the virtual mouse using:
```powershell
py gesture_mouse.py
```

### Supported Gestures
Move your hand inside the **white touchpad boundary box** on the camera screen. The cursor follows your index finger tip smoothly.

| Mode | Gesture | Mouse Action | HUD Color |
| :--- | :--- | :--- | :--- |
| **Move (Hover)** | **Index finger UP** (others folded) | Move mouse cursor | **Green** |
| **Left Click / Drag** | **Index + Thumb pinched together** | Click & Hold (Drag) | **Red** |
| **Right Click** | **Index + Pinky fingers UP** (others folded) | Single Right Click | **Yellow** |
| **Scroll Mode** | **Index + Middle fingers UP** (separated) | Slide hand up/down to scroll | **Blue** |

* **To Stop the App**: Click on the camera window and press **`q`** on your keyboard, or press **`Ctrl` + `C`** in your terminal.

---

## 🏋️ How to Train the AI on Your Own Hand

Training the model on your own hand takes **less than 2 minutes** and ensures the mouse works perfectly with your camera angle and hand size.

### 1. Start Fresh (Optional)
If you want to record a brand-new dataset, delete the existing data file first:
```powershell
Remove-Item gesture_data.csv -ErrorAction SilentlyContinue
```

### 2. Run the Data Collector
```powershell
py collect_data.py
```

### 3. Record Poses
Hold your hand in front of the camera and **press and hold** the corresponding key for 5–10 seconds per pose while moving your hand slightly to collect different distances and angles (aim for **150–200 frames** per class):
* Hold **Index finger UP** ➔ Hold **`0`**
* Hold **Index + Thumb pinch** ➔ Hold **`1`**
* Hold **Index + Pinky UP** ➔ Hold **`2`**
* Hold **Index + Middle UP** ➔ Hold **`3`**

Press **`q`** in the window when done. This saves the training data to `gesture_data.csv`.

### 4. Train the Model
```powershell
py train_model.py
```
This script trains a Random Forest Classifier and an MLP Neural Network, compares their accuracies, and saves the best model to **`gesture_model.pkl`**.

---

## 🧹 How to Reset or Delete the Model

If you want to clear your custom-trained model or dataset to start completely from scratch, run this command in your terminal:

```powershell
Remove-Item gesture_model.pkl, gesture_data.csv -ErrorAction SilentlyContinue
```

Alternatively, you can manually delete these two files from your file explorer:
* **`gesture_model.pkl`** (the compiled AI model)
* **`gesture_data.csv`** (your recorded coordinate coordinates)
