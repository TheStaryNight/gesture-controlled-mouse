# Gesture Controlled Virtual Mouse (Custom ML Edition)

A real-time computer vision application that allows you to control your PC's mouse cursor, perform left clicks/drags, right clicks, and scroll using hand gestures captured from your laptop's built-in webcam. 

This project uses **Google MediaPipe Tasks API** for high-precision hand landmark extraction and a **Custom Random Forest Classifier** (via Scikit-Learn) trained specifically on *your* hand for 100% accuracy and zero jitter.

> [!IMPORTANT]
> **Trained model binaries (`.pkl`) and datasets (`.csv`) are NOT pushed to GitHub.** 
> Anyone cloning this repository **must train the model on their own hand first** before they can run the virtual mouse!

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/TheStaryNight/gesture-controlled-mouse.git
   cd gesture-controlled-mouse
   ```

2. **Install the dependencies**:
   Make sure you have Python 3.12+ installed, then run:
   ```powershell
   py -m pip install -r requirements.txt
   ```
   *This installs `opencv-python`, `mediapipe`, `pynput`, `screeninfo`, `scikit-learn`, and `streamlit`.*

---

## Dashboard (Streamlit UI)

This project includes a web-based dashboard that lets you do everything from a single interface — no need to remember terminal commands.

**To launch the dashboard:**
```powershell
py -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

From the dashboard you can:
- **Collect Data** — opens the webcam collector window
- **Train Model** — trains and auto-selects the best classifier
- **Run Mouse** — launches the gesture-controlled virtual mouse
- **Delete Model / Dataset** — reset everything to start fresh

The dashboard also shows your current dataset stats and model status at a glance.

> [!TIP]
> If you prefer using the terminal instead of the dashboard, follow the manual steps below.

---

## 🏋️ Step 1: Train the AI on Your Own Hand (REQUIRED FIRST STEP)

Training the model on your own hand takes **less than 2 minutes** and ensures the mouse works perfectly with your hand size and camera angle.

### 1. Run the Data Collector
Start the webcam collection script:
```powershell
py collect_data.py
```

### 2. Record Your Gestures
Hold your hand in front of the camera and **press and hold** the corresponding number key on your keyboard for 5–10 seconds per pose (aim for **150–200 frames** per class). Move and tilt your hand slightly while recording to capture variations:
* Hold **Index finger UP** (others folded) ➔ Hold **`0`** (Move Cursor)
* Hold **Index + Thumb both UP** ➔ Hold **`1`** (Left Click / Drag)
* Hold **Index + Pinky UP** (others folded) ➔ Hold **`2`** (Right Click)
* Hold **Index + Middle UP** (separated) ➔ Hold **`3`** (Scroll Mode)

Press **`q`** on the camera window when done. This saves the coordinates to `gesture_data.csv`.

### 3. Train the Model
```powershell
py train_model.py
```
This script trains a Random Forest Classifier and an MLP Neural Network, compares their accuracies, and saves the best model to **`gesture_model.pkl`**.

---

## 🎮 Step 2: Run the Virtual Mouse

Once you have successfully trained your model and generated `gesture_model.pkl`, start the virtual mouse using:
```powershell
py gesture_mouse.py
```

### Supported Gestures
Move your hand inside the **white touchpad boundary box** on the camera screen. The cursor follows your index finger tip smoothly.

| Mode | Gesture | Mouse Action | HUD Color |
| :--- | :--- | :--- | :--- |
| **Move (Hover)** | **Index finger UP** (others folded) | Move mouse cursor | **Green** |
| **Left Click / Drag** | **Index + Thumb both UP** | Click & Hold (Drag) | **Red** |
| **Right Click** | **Index + Pinky fingers UP** (others folded) | Single Right Click | **Yellow** |
| **Scroll Mode** | **Index + Middle fingers UP** (separated) | Slide hand up/down to scroll | **Blue** |

* **To Stop the App**: Click on the camera window and press **`q`** on your keyboard, or press **`Ctrl` + `C`** in your terminal.

---

## 🧹 How to Reset or Delete the Model

If you want to clear your custom-trained model or dataset to start completely from scratch, run this command in your terminal:

```powershell
Remove-Item gesture_model.pkl, gesture_data.csv -ErrorAction SilentlyContinue
```

Alternatively, you can manually delete these two files from your file explorer:
* **`gesture_model.pkl`** (the compiled AI model)
* **`gesture_data.csv`** (your recorded coordinate data)

Or use the **Delete** buttons in the Streamlit dashboard.

