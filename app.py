import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np
from PIL import Image
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# -------------------------
# GLOBAL STATE FOR HISTORY
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []


class BasuraVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_pred = "Starting..."
        self.model_loaded = False
        self.model = None
        self.frame_count = 0
        self.labels = ["biodegradable", "recyclable", "residual"]

        self.load_model()

    def load_model(self):
        if not TF_AVAILABLE:
            self.last_pred = "TF not available"
            return

        model_path = "models/basuranet_final.h5"

        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                self.last_pred = "Model loaded!"
                print(f"✅ Model loaded from {model_path}")
            else:
                self.last_pred = f"Model not found at {model_path}"
                print(f"❌ Model not found at {model_path}")
        except Exception as e:
            self.last_pred = f"Load error"
            print(f"❌ Error loading model: {e}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        if self.frame_count % 20 == 0:
            if self.model_loaded and self.model is not None:
                try:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    input_img = cv2.resize(rgb, (224, 224))
                    input_array = np.expand_dims(input_img / 255.0, axis=0)

                    predictions = self.model.predict(input_array, verbose=0)[0]
                    class_idx = np.argmax(predictions)
                    confidence = predictions[class_idx]

                    if class_idx < len(self.labels):
                        label = self.labels[class_idx]
                    else:
                        label = f"Class {class_idx}"

                    if confidence > 0.5:
                        result = f"{label} ({confidence:.2f})"
                    else:
                        result = "Uncertain"

                    self.last_pred = result

                    # ADD TO HISTORY
                    st.session_state.history.append(result)

                except:
                    self.last_pred = "Pred error"

        cv2.putText(
            img,
            self.last_pred,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------
# HEADER DESIGN
# -------------------------

st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(90deg, #0073e6, #00c896);
        padding: 18px 0;
        text-align: center;
        border-radius: 6px;
        margin-bottom: 20px;
        color: white;
        font-family: Arial, sans-serif;
    }
    .header-title {
        font-size: 34px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .header-sub {
        font-size: 16px;
        margin-top: 5px;
        opacity: 0.9;
    }
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
# LIVE CAMERA
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
# HISTORY SECTION
# -------------------------

st.write("### 🧾 Prediction History")

if len(st.session_state.history) == 0:
    st.info("No predictions yet...")
else:
    for i, h in enumerate(st.session_state.history[-20:][::-1], start=1):
        st.write(f"{i}. **{h}**")

if st.button("Clear History"):
    st.session_state.history = []
    st.experimental_rerun()
