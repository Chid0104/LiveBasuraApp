# app.py — BasuraNet Premium (3-class) — safe prediction text (no ???)
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np
import os
import traceback

st.set_page_config(page_title="BasuraNet — Premium", page_icon="🗑️", layout="wide")

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

st.markdown(
    """
    <div class="header">
      <div>
        <div class="title">BasuraNet</div>
        <div class="subtitle">Live Waste Classification</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

class PremiumVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.frame_count = 0
        self.last_label = ""
        self.last_conf = 0.0
        self.last_pred_text = ""  # sanitized display string
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
        if preds is None:
            return None
        if preds.ndim == 2 and preds.shape[0] == 1:
            preds = preds[0]
        return preds

    def _format_prediction(self, preds):
        # preds must be a 1D array of floats (probabilities or logits)
        try:
            if preds is None:
                return ""
            preds = np.asarray(preds, dtype=float)
            if preds.size == 0 or not np.isfinite(preds).all():
                return ""
            # If values look like logits (not normalized), try softmax safely
            if preds.min() < 0 or preds.max() > 1 or not np.isclose(preds.sum(), 1.0, atol=1e-2):
                exps = np.exp(preds - np.max(preds))
                probs = exps / (np.sum(exps) + 1e-12)
            else:
                probs = preds / (np.sum(preds) + 1e-12)

            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if not np.isfinite(conf) or conf < 0.0 or conf > 1.0:
                return ""
            label = self.labels[idx] if 0 <= idx < len(self.labels) else "Unknown"
            return f"{label} — {conf*100:.1f}%"
        except Exception:
            return ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Only run prediction periodically to reduce load
        if self.model_loaded and (self.frame_count % 10 == 0):
            try:
                batch = self.preprocess(img)
                preds = self.predict(batch)
                safe_text = self._format_prediction(preds)
                # If formatting returned empty, do not show "???" or invalid text
                if safe_text:
                    self.last_pred_text = safe_text
                else:
                    # keep previous text or clear (choose clear to avoid stale invalids)
                    self.last_pred_text = ""
            except Exception:
                # on any error, make sure we don't show invalid placeholder
                self.last_pred_text = ""

        # Overlay: clean dark green rectangle and text only if we have text
        h, w, _ = img.shape
        box_w = min(520, int(w * 0.6))
        box_h = 56
        margin = 14
        x1, y1 = margin, h - box_h - margin
        x2, y2 = x1 + box_w, y1 + box_h

        sub = img[y1:y2, x1:x2]
        if sub.size != 0:
            overlay = sub.copy()
            overlay[:] = (25, 70, 40)  # solid dark green
            cv2.addWeighted(overlay, 0.75, sub, 0.25, 0, sub)
            img[y1:y2, x1:x2] = sub

        if self.last_pred_text:
            cv2.putText(img, self.last_pred_text, (x1 + 18, y1 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Live Camera")

ctx = webrtc_streamer(
    key="premium_cam",
    video_processor_factory=PremiumVideoProcessor,
    media_stream_constraints={"video": {"width": 720, "height": 720}, "audio": False},
    async_processing=True,
)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.caption("BasuraNet — Premium real-time waste detection. 3-class AI model.")
