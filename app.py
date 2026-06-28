import streamlit as st
import subprocess
import os
import sys
import csv

# ============================================================
# Configuration — all paths relative to THIS script's folder
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "gesture_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "gesture_model.pkl")
COLLECT_SCRIPT = os.path.join(BASE_DIR, "collect_data.py")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_model.py")
MOUSE_SCRIPT = os.path.join(BASE_DIR, "gesture_mouse.py")

CLASSES = {
    0: "Move — Index Up",
    1: "Left Click — Index + Thumb Up",
    2: "Right Click — Index + Pinky Up",
    3: "Scroll — Index + Middle Up"
}

GESTURE_COLORS = {
    0: "#4ade80",  # green
    1: "#f87171",  # red
    2: "#fbbf24",  # yellow
    3: "#60a5fa",  # blue
}

# ============================================================
# Helpers
# ============================================================
def get_dataset_stats():
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) == 43:
                    try:
                        label = int(row[-1])
                        if label in counts:
                            counts[label] += 1
                            total += 1
                    except ValueError:
                        pass
    return counts, total

def model_exists():
    return os.path.exists(MODEL_FILE)

def dataset_exists():
    return os.path.exists(CSV_FILE)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Gesture Controlled Mouse",
    page_icon="cursor",
    layout="wide"
)

# ============================================================
# Inject CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Hero ---- */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.2rem;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* ---- Stat card ---- */
    .stat-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
    }
    .stat-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .stat-value.ok { color: #4ade80; }
    .stat-value.bad { color: #f87171; }
    .stat-value.warn { color: #fbbf24; }

    /* ---- Gesture row ---- */
    .g-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        margin-bottom: 0.45rem;
    }
    .g-row .g-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .g-row .g-name {
        color: #cbd5e1;
        font-size: 0.85rem;
        font-weight: 400;
        flex: 1;
    }
    .g-row .g-count {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 600;
        margin-left: 12px;
    }

    /* ---- Section header ---- */
    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.8rem;
        margin-top: 1.5rem;
    }

    /* ---- Divider ---- */
    .divider {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 1.8rem 0;
    }

    /* ---- Guide steps ---- */
    .guide-step {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.5rem;
    }
    .guide-step .step-num {
        color: #667eea;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 0.2rem;
    }
    .guide-step .step-text {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.45;
    }

    /* ---- Footer ---- */
    .footer-text {
        text-align: center;
        color: #334155;
        font-size: 0.75rem;
        padding: 1rem 0;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="hero">
    <h1>Gesture Controlled Mouse</h1>
    <p>Real-time hand tracking with MediaPipe + ML classification</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Status Cards
# ============================================================
counts, total_samples = get_dataset_stats()
has_model = model_exists()
has_dataset = dataset_exists()

col1, col2, col3 = st.columns(3)

with col1:
    val_class = "ok" if has_dataset and total_samples > 0 else "bad"
    val_text = f"{total_samples:,}" if has_dataset else "—"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Dataset Samples</div>
        <div class="stat-value {val_class}">{val_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    m_class = "ok" if has_model else "bad"
    m_text = "Trained" if has_model else "Not trained"
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Model Status</div>
        <div class="stat-value {m_class}">{m_text}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    g_count = sum(1 for c in counts.values() if c > 0)
    g_class = "ok" if g_count == 4 else ("warn" if g_count > 0 else "bad")
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Classes Recorded</div>
        <div class="stat-value {g_class}">{g_count} / 4</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Gesture Breakdown — render each row separately to avoid bugs
# ============================================================
if has_dataset and total_samples > 0:
    st.markdown('<div class="section-title">Samples per gesture</div>', unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2)
    for k, v in CLASSES.items():
        target_col = g_col1 if k < 2 else g_col2
        with target_col:
            color = GESTURE_COLORS[k]
            st.markdown(f"""
            <div class="g-row">
                <span class="g-dot" style="background:{color};"></span>
                <span class="g-name">{v}</span>
                <span class="g-count">{counts[k]}</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ============================================================
# Actions
# ============================================================
st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)

a1, a2, a3, a4 = st.columns(4)

with a1:
    st.markdown("**Collect Data**")
    st.caption("Record hand gesture samples via webcam.")
    if st.button("Start Collection", use_container_width=True, type="primary"):
        with st.spinner("Webcam collector running — close the OpenCV window to return."):
            result = subprocess.run(
                [sys.executable, COLLECT_SCRIPT],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("Collection finished.")
            else:
                st.error("Error during collection.")
                st.code(result.stderr[-400:] if result.stderr else "Unknown error.")
        st.rerun()

with a2:
    st.markdown("**Train Model**")
    st.caption("Train RF + MLP, auto-select best model.")
    train_off = not has_dataset or total_samples < 10
    if st.button("Train Model", use_container_width=True, type="primary", disabled=train_off):
        with st.spinner("Training in progress..."):
            result = subprocess.run(
                [sys.executable, TRAIN_SCRIPT],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("Training complete.")
                st.code(result.stdout[-800:] if result.stdout else "Done.")
            else:
                st.error("Training failed.")
                st.code(result.stderr[-400:] if result.stderr else "Unknown error.")
        st.rerun()
    if train_off:
        st.caption("Requires at least 10 samples.")

with a3:
    st.markdown("**Run Mouse**")
    st.caption("Launch gesture-controlled virtual mouse.")
    mouse_off = not has_model
    if st.button("Start Mouse", use_container_width=True, type="primary", disabled=mouse_off):
        with st.spinner("Gesture mouse active — press 'q' in the OpenCV window to stop."):
            result = subprocess.run(
                [sys.executable, MOUSE_SCRIPT],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("Mouse stopped.")
            else:
                st.error("Error during mouse control.")
                st.code(result.stderr[-400:] if result.stderr else "Unknown error.")
    if mouse_off:
        st.caption("Train a model first.")

with a4:
    st.markdown("**Manage Data**")
    st.caption("Delete model or dataset to reset.")

    if st.button("Delete Model", use_container_width=True, disabled=not has_model):
        os.remove(MODEL_FILE)
        st.success("Model deleted.")
        st.rerun()

    if st.button("Delete Dataset", use_container_width=True, disabled=not has_dataset):
        os.remove(CSV_FILE)
        st.success("Dataset deleted.")
        st.rerun()

    if st.button("Delete All", use_container_width=True, disabled=not (has_model or has_dataset)):
        if has_model:
            os.remove(MODEL_FILE)
        if has_dataset:
            os.remove(CSV_FILE)
        st.success("All data deleted.")
        st.rerun()

# ============================================================
# Guide
# ============================================================
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Quick guide</div>', unsafe_allow_html=True)

st.markdown("""
<div class="guide-step">
    <div class="step-num">STEP 1 — COLLECT</div>
    <div class="step-text">Click <b>Start Collection</b>. A webcam window opens. Hold your gesture and press the key (0–3) to record frames. Aim for 150+ per gesture. Press <b>q</b> to stop.</div>
</div>
<div class="guide-step">
    <div class="step-num">STEP 2 — TRAIN</div>
    <div class="step-text">Click <b>Train Model</b>. Trains both Random Forest and MLP, compares accuracy, and saves the best one automatically.</div>
</div>
<div class="guide-step">
    <div class="step-num">STEP 3 — RUN</div>
    <div class="step-text">Click <b>Start Mouse</b>. Your hand now controls the cursor in real-time. Press <b>q</b> to quit.</div>
</div>
<div class="guide-step">
    <div class="step-num">RETRAIN</div>
    <div class="step-text">Collect more data (it appends to existing) and click Train again. Or delete everything to start fresh.</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="footer-text">OpenCV · MediaPipe · Scikit-Learn · Streamlit</div>', unsafe_allow_html=True)
