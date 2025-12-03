# app.py — Premium UI for BasuraNet (3-class)
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
    /* Page background */
    .stApp {
        background: linear-gradient(180deg,#f6f8fb 0%, #ffffff 100%);
    }
    /* Header */
    .header {
        background: linear-gradient(90deg,#0f172a,#0b3d2e);
        padding: 18px 28px;
        border-radius: 12px;
        color: #fff;
        box-shadow: 0 8px 30px rgba(12,18,27,0.08);
        margin-bottom: 18px;
    }
    .title {
        font-size:28px;
        font-weight:700;
        margin:0;
        letter-spacing:0.6px;
    }
    .subtitle {
        font-size:13px;
        color: #d2f3e6;
        margin-top:4px;
        opacity:0.95;
    }

    /* Camera card */
    .card {
        background: #ffffff;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 10px 40px rgba(18,38,60,0.06);
        border: 1px solid rgba(15,23,42,0.03);
    }

    /* Right panel items */
    .stat {
        background: linear-gradient(180deg,#ffffff,#fbfdff);
        border-radius: 10px;
        padding: 12px;
        margin-bottom:10px;
        border:1px solid #eef4f8;
    }

    /* Small muted text */
    .muted { color:#68707d; font-size:13px; }

    /* Button styling */
    .streamlit-button {
        border-radius:8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    f"""
    <div class="header" style="display:flex;align-items:center;justify-content:space-between">
      <div>
        <div class="title">BasuraNet</div>
        <div class="subtitle">Premium Live Waste Classification</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:12px;color:#cdebd8">Model:</div>
        <div style="font-weight:600;color:#fff">basuranet_final.h5</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar (controls & info) ----------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    cam_col = st.radio("Camera", options=["Front", "Back"], index=0, horizontal=True)
    # map to facingMode
    facing_mode = "user" if cam_col == "Front" else "environment"
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "BasuraNet classifies live video frames into **Biodegradable**, **Recyclable**, or **Residual**.\n\n"
        "- Predictions occur every few frames to keep UI smooth.\n- Model file: `models/basuranet_final.h5`"
    )
    st.markdown("---")
    st.markdown("### ⚠️ Notes")
    st.markdown("<span class='muted'>Ensure model file exists and TensorFlow is installed on the host.</span>", unsafe_allow_html=True)


# ---------- Video Processor (keeps logic robust) ----------
class PremiumVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.frame_count = 0

        # visible state for UI
        self.last_label = "Loading..."
        self.last_conf = 0.0
        self.last_pred_text = "Loading..."

        # labels (3-class)
        self.labels = ["Biodegradable", "Recyclable", "Residual"]

        # attempt model load
        self._try_load_model()

    def _try_load_model(self):
        try:
            import tensorflow as tf
            model_path = "models/basuranet_final.h5"
            if os.path.exists(model_path):
                # load without compiling (safer)
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model_loaded = True
                self.last_label = "Model Ready"
                self.last_pred_text = "Model Ready"
            else:
                self.last_label = "Model file missing"
                self.last_pred_text = "Model file missing"
                self.model_loaded = False
        except Exception:
            self.model_loaded = False
            self.last_label = "TF load error"
            self.last_pred_text = "TF load error"
            traceback.print_exc()

    def preprocess(self, img_bgr):
        # BGR -> RGB : resize -> normalize
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

        # control prediction frequency to reduce lag
        PRED_INTERVAL = 10

        if self.model_loaded and (self.frame_count % PRED_INTERVAL == 0):
            try:
                batch = self.preprocess(img)
                preds = self.predict(batch)

                idx = int(np.argmax(preds))
                conf = float(preds[idx])

                # map to labels safely
                label = self.labels[idx] if 0 <= idx < len(self.labels) else f"class_{idx}"
                self.last_label = label
                self.last_conf = conf
                self.last_pred_text = f"{label} — {conf*100:.1f}%"
            except Exception:
                self.last_label = "Pred Error"
                self.last_conf = 0.0
                self.last_pred_text = "Pred Error"
                traceback.print_exc()

        # draw lower-third modern overlay
        overlay_text = self.last_pred_text
        h, w, _ = img.shape
        box_w = min(520, int(w * 0.6))
        box_h = 56
        margin = 14
        x1 = margin
        y1 = h - box_h - margin
        x2 = x1 + box_w
        y2 = y1 + box_h

        # semi-transparent rounded-like rectangle (simple filled)
        sub = img[y1:y2, x1:x2]
        if sub.size != 0:
            overlay = sub.copy()
            overlay[:] = (18, 52, 38)  # dark green background
            cv2.addWeighted(overlay, 0.68, sub, 0.32, 0, sub)
            img[y1:y2, x1:x2] = sub

        # text
        cv2.putText(img, overlay_text, (x1 + 18, y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- Main layout: Camera card + Right stats ----------
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Live Camera")
    # Video streamer: use facingMode control from sidebar
    ctx = webrtc_streamer(
        key="basura_premium",
        video_processor_factory=PremiumVideoProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": facing_mode,
                "width": 640,
                "height": 480
            },
            "audio": False
        },
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='stat'>", unsafe_allow_html=True)
    st.markdown("### Current Prediction")
    # placeholders to be updated from video processor
    pred_label = st.empty()
    pred_conf = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='stat'>", unsafe_allow_html=True)
    st.markdown("### Classes")
    st.markdown("- **Biodegradable** — Food/organic waste")
    st.markdown("- **Recyclable** — Plastic, metal, paper")
    st.markdown("- **Residual** — Mixed/non-recyclable")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='stat'>", unsafe_allow_html=True)
    st.markdown("### Actions")
    if st.button("Take Snapshot"):
        # snapshot: grab last frame from ctx (if available)
        try:
            if ctx and ctx.video_transformer and ctx.video_transformer.frame_buffer:
                # don't rely on internals; best-effort
                st.info("Snapshot feature not implemented in this build.")
            else:
                st.info("Snapshot not available.")
        except Exception:
            st.info("Snapshot not available.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- UI updater: read latest from processor ----------
def update_ui_from_processor(context):
    if not context:
        return
    processor = context.video_processor
    if not processor:
        return
    # read safely
    label = getattr(processor, "last_label", "—")
    conf = getattr(processor, "last_conf", 0.0)
    pred_label.markdown(f"**{label}**")
    # render a nice progress bar
    pred_conf.progress(min(max(conf, 0.0), 1.0))

# call updater every rerun
update_ui_from_processor(ctx)

# ---------- Footer ----------
st.markdown("---")
st.caption("Premium UI — BasuraNet. Keep camera steady for best results.")
