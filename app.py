# app.py — Premium BasuraNet (modern colors, 3-class)
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np
import os
import traceback

# ---------- Config ----------
st.set_page_config(page_title="BasuraNet — Premium", page_icon="🗑️", layout="wide")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#f0f2f5 0%, #ffffff 100%); }

    .header {
        background: linear-gradient(90deg,#1f2937,#4ade80);
        padding: 18px 28px;
        border-radius: 12px;
        color: #fff;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        margin-bottom: 18px;
        display:flex;
        align-items:center;
    }
    .title { font-size:28px; font-weight:700; letter-spacing:0.6px; }
    .subtitle { font-size:13px; color:#d1fae5; opacity:0.95; }

    .card {
        background:#ffffff;
        border-radius:14px;
        padding:14px;
        box-shadow:0 10px 40px rgba(0,0,0,0.06);
        border:1px solid rgba(15,23,42,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    f"""
    <div class="header">
      <div>
        <div class="title">BasuraNet</div>
        <div class="subtitle">Live Waste Classification</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Video Processor ----------
class PremiumVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.frame_count = 0
        self.last_label = "Loading..."
        self.last_conf = 0.0
        self.last_pred_text = "Loading..."
        self.labels = ["Biodegradable", "Recyclable", "Residual"]
        self._try_load_model()

    def _try_load_model(self):
        try:
            import tensorflow as tf
            model_path = "models/basuranet_final.h5"
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model_loaded = True
                self.last_pred_text = "Model Ready"
            else:
                self.last_pred_text = "Model file missing"
        except Exception:
            self.last_pred_text = "TF load error"
            traceback.print_exc()

    def preprocess(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, batch):
        preds = self.model.predict(batch, verbose=0)
        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]
        return preds

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.model_loaded and (self.frame_count % 10 == 0):
            try:
                batch = self.preprocess(img)
                preds = self.predict(batch)
                idx = int(np.argmax(preds))
                conf = float(preds[idx])
                label = self.labels[idx]
                self.last_label = label
                self.last_conf = conf
                self.last_pred_text = f"{label} — {conf*100:.1f}%"
            except:
                self.last_pred_text = "Prediction Error"

        # Overlay with modern gradient
        h, w, _ = img.shape
        box_w = min(520, int(w * 0.6))
        box_h = 56
        margin = 14
        x1, y1 = margin, h - box_h - margin
        x2, y2 = x1 + box_w, y1 + box_h

        sub = img[y1:y2, x1:x2]
        if sub.size != 0:
            overlay = sub.copy()
            # gradient background
            for i in range(box_h):
                alpha = i / box_h
                overlay[i, :] = (0 + int(120*alpha), 128 + int(50*alpha), 64 + int(60*alpha))
            cv2.addWeighted(overlay, 0.75, sub, 0.25, 0, sub)
            img[y1:y2, x1:x2] = sub

        # Text
        cv2.putText(img, self.last_pred_text, (x1 + 18, y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- Main layout ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Live Camera")
ctx = webrtc_streamer(
    key="premium_cam",
    video_processor_factory=PremiumVideoProcessor,
    media_stream_constraints={"video": {"width": 720, "height": 720}, "audio": False},
    async_processing=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("BasuraNet — Premium real-time waste detection. 3-class AI model.")
