import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False


# =============================
# PAGE & STYLE
# =============================
st.set_page_config(page_title="BasuraNet", page_icon="🗑️", layout="wide")


# =============================
# CAMERA MODE
# =============================
if "camera_mode" not in st.session_state:
    st.session_state.camera_mode = "user"   # user = front cam


# =============================
# HEADER
# =============================
st.markdown("""
<h1 style="text-align:center; font-size:42px; font-weight:600; margin-bottom:0;">
BasuraNet
</h1>
<p style="text-align:center; color:#666; margin-top:2px;">
Real-Time Trash Classification
</p>
<hr style="width:160px; margin:auto; border:1px solid #1abc9c;">
""", unsafe_allow_html=True)


# =============================
# MODEL PROCESSOR (QUALITY FIX)
# =============================
class BasuraVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_pred = "Loading..."
        self.frame_count = 0
        self.model_loaded = False
        
        self.labels = ["Biodegradable", "Recyclable", "Residual"]
        self.load_model()


    def load_model(self):
        if not TF_AVAILABLE:
            self.last_pred = "TensorFlow missing"
            return

        try:
            self.model = tf.keras.models.load_model("models/basuranet_final.h5")
            self.model_loaded = True
            self.last_pred = "Model loaded"
        except:
            self.last_pred = "Model load error"


    def predict_class(self, img):

        # --- High quality preprocessing ---
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)[0]
        idx = int(np.argmax(pred))
        return self.labels[idx], float(pred[idx])


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Predict every 10 frames (quality + speed balance)
        if self.model_loaded and self.frame_count % 10 == 0:
            try:
                label, conf = self.predict_class(img)
                self.last_pred = f"{label} — {conf*100:.1f}%"
            except:
                self.last_pred = "Prediction error"

        # =============================
        # MODERN OVERLAY (CLEAN UI)
        # =============================
        overlay_color = (22, 160, 133)   # teal

        cv2.rectangle(
            img,
            (10, 10), (450, 60),
            overlay_color, -1
        )

        cv2.putText(
            img,
            self.last_pred,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
            cv2.LINE_AA              # <-- Anti aliasing for cleaner text
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")



# =============================
# CAMERA SWITCH BUTTON
# =============================
colA, colB = st.columns([1,1])

with colA:
    if st.button("🔄 Switch Camera"):
        st.session_state.camera_mode = (
            "environment" if st.session_state.camera_mode == "user" else "user"
        )

with colB:
    st.write(
        "**Camera:** Front**" if st.session_state.camera_mode == "user"
        else "**Camera:** Back**"
    )


# =============================
# LIVE CAMERA
# =============================
st.markdown("""
<div style="
    background:#ffffff; 
    padding:20px;
    border-radius:16px;
    border:1px solid #e5e5e5;
    box-shadow:0 6px 22px rgba(0,0,0,0.09);
    margin-top:14px;">
""", unsafe_allow_html=True)


# ⚡ FIXED QUALITY SETTINGS
webrtc_streamer(
    key="basura_quality",
    video_processor_factory=BasuraVideoProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": st.session_state.camera_mode,
            "width": 640,          # Higher resolution
            "height": 480
        },
        "audio": False
    },
)


st.markdown("</div>", unsafe_allow_html=True)
st.caption("Hold a waste item in front of the camera.")
