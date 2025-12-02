import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np
import os
from collections import deque

# Optional: enable detailed TF error printing to console
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    print("TF import error:", e)

# -------------------------
# GLOBAL STATE FOR HISTORY
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Module-level buffer used by the video thread to send predictions to main thread
PRED_HISTORY = deque(maxlen=500)  # thread-shared buffer (append-only from recv)

# -------------------------
# VIDEO PROCESSOR (prediction logic unchanged)
# -------------------------
class BasuraVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_pred = "Starting..."
        self.model_loaded = False
        self.model = None
        self.frame_count = 0

        # IMPORTANT: make labels match the model's output size (4 classes)
        self.labels = ["biodegradable", "recyclable", "residual", "not_trash"]

        # Try to load model right away
        self.load_model()

    def load_model(self):
        """Load the .h5 model safely."""
        if not TF_AVAILABLE:
            self.last_pred = "TF not available"
            print("TF not available in environment.")
            return

        model_path = "models/basuranet_final.h5"

        try:
            if os.path.exists(model_path):
                # load without compiling to avoid issues with custom losses/metrics
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model_loaded = True
                self.last_pred = "Model loaded!"
                print(f"✅ Model loaded from {model_path}")
            else:
                self.last_pred = f"Model not found at {model_path}"
                print(f"❌ Model not found at {model_path}")
        except Exception as e:
            self.model_loaded = False
            self.model = None
            self.last_pred = "Load error"
            print("❌ Error loading model:", repr(e))

    def recv(self, frame):
        """Called in the webrtc background thread. Keep this efficient and thread-safe."""
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Do inference every N frames to reduce load
        if self.frame_count % 20 == 0:
            if not self.model_loaded:
                # attempt to load model if it wasn't available earlier
                try:
                    self.load_model()
                except Exception as e:
                    print("Error while attempting load_model in recv:", repr(e))

            if self.model_loaded and self.model is not None:
                try:
                    # Preprocess
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    input_img = cv2.resize(rgb, (224, 224))
                    input_array = np.expand_dims(input_img / 255.0, axis=0)

                    # Predict
                    preds = self.model.predict(input_array, verbose=0)
                    if preds is None:
                        raise RuntimeError("Model returned None on predict()")

                    # handle shape (ensure 1D array)
                    preds = np.asarray(preds)
                    if preds.ndim == 2 and preds.shape[0] == 1:
                        preds = preds[0]

                    class_idx = int(np.argmax(preds))
                    confidence = float(preds[class_idx])

                    # Safely map label (if label index out of range, produce fallback label)
                    if 0 <= class_idx < len(self.labels):
                        label = self.labels[class_idx]
                    else:
                        label = f"Class_{class_idx}"

                    if confidence > 0.5:
                        result = f"{label} ({confidence:.2f})"
                    else:
                        result = "Uncertain"

                    self.last_pred = result

                    # Append to module-level buffer (thread-safe enough for append-only use)
                    try:
                        PRED_HISTORY.append(result)
                    except Exception as e:
                        # keep running even if buffer append fails
                        print("Warning: failed to append to PRED_HISTORY:", repr(e))

                except Exception as e:
                    # Surface the full exception to the worker log for debugging
                    print("Prediction error in recv():", repr(e))
                    self.last_pred = "Pred error"

        # draw overlay text
        cv2.putText(
            img,
            self.last_pred,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Frame: {self.frame_count}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------
# HEADER (UI only)
# -------------------------
st.set_page_config(page_title="BasuraNet", page_icon="♻️", layout="centered")

st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(90deg, #0073e6, #00c896);
        padding: 18px 0;
        text-align: center;
        border-radius: 6px;
        margin-bottom: 12px;
        color: white;
        font-family: Arial, sans-serif;
    }
    .header-title { font-size: 30px; font-weight: 700; margin: 0; }
    .header-sub { font-size: 14px; margin-top: 4px; opacity: 0.95; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-box">
        <div class="header-title">🗑️ BasuraNet</div>
        <div class="header-sub">Real-Time Waste Classification</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Hold an item in front of the camera and wait for prediction.")

# -------------------------
# Start camera streamer
# -------------------------
webrtc_streamer(
    key="basuranet",
    video_processor_factory=BasuraVideoProcessor,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    },
)

# -------------------------
# Pull predictions from PRED_HISTORY into st.session_state.history
# (This runs in the main Streamlit thread and is safer than modifying session_state inside recv())
# -------------------------
# Move up to 100 new items into session_state.history each rerun
moved = 0
while PRED_HISTORY and moved < 100:
    try:
        st.session_state.history.append(PRED_HISTORY.popleft())
    except Exception as e:
        print("Error moving item from PRED_HISTORY to session_state:", repr(e))
        break
    moved += 1

# -------------------------
# HISTORY UI
# -------------------------
st.write("### 🧾 Prediction History")
if len(st.session_state.history) == 0:
    st.info("No predictions yet...")
else:
    # show last 20, newest first
    for i, h in enumerate(st.session_state.history[-20:][::-1], start=1):
        st.write(f"{i}. **{h}**")

if st.button("Clear History"):
    st.session_state.history = []
    st.experimental_rerun()
