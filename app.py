import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False


# =============================
#   HEADER
# =============================
st.markdown(
    """
    <div style="text-align:center; padding: 15px; background:#0D0D0D; border-bottom:3px solid #4CAF50;">
        <h1 style="color:#4CAF50; font-size:40px; margin:0;">BasuraNet CCTV</h1>
        <p style="color:#BBBBBB; font-size:15px; margin-top:6px;">
            Automated Waste Monitoring Camera
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================
#   VIDEO PROCESSOR
# =============================
class BasuraVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.frame_count = 0

        self.last_pred = "Initializing..."
        self.labels = ["biodegradable", "recyclable", "residual"]

        self.history = []
        self.load_model()


    def load_model(self):
        if not TF_AVAILABLE:
            return

        model_path = "models/basuranet_final.h5"

        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
            self.last_pred = "Model Ready"
        except:
            self.last_pred = "Model Load Error"


    def predict_class(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224)).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = self.model.predict(img, verbose=0)[0]

        idx = int(np.argmax(preds))
        conf = float(preds[idx])

        return self.labels[idx], conf


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.model_loaded and self.frame_count % 10 == 0:
            try:
                label, conf = self.predict_class(img)
                self.last_pred = f"{label} ({conf:.2f})"

                self.history.append(self.last_pred)
                if len(self.history) > 8:
                    self.history.pop(0)

            except:
                self.last_pred = "Prediction Error"

        # ------------------------------------------------------
        # MODERN OVERLAY STYLE (professional CCTV style)
        # ------------------------------------------------------

        overlay_text = self.last_pred

        # Background rectangle (semi transparent)
        (w, h), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (10, 10), (10 + w + 20, 10 + h + 20), (0, 0, 0, 0.4), -1)

        # Text
        cv2.putText(
            img,
            overlay_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")



# =============================
#   STREAMLIT LAYOUT
# =============================
col1, col2 = st.columns([5,2])

with col1:
    ctx = webrtc_streamer(
        key="basuranet",
        video_processor_factory=BasuraVideoProcessor,
        media_stream_constraints={
            "video": {"width": 960, "height": 540},
            "audio": False
        },
    )

with col2:
    st.markdown(
        "<h3 style='color:#4CAF50; text-align:center;'>Activity Log</h3>",
        unsafe_allow_html=True
    )

    history_box = st.empty()


def update_history(processor):
    if processor and hasattr(processor, "history"):
        items = processor.history
        html = "<br>".join([f"<div style='padding:4px; color:#DDD;'>{x}</div>" for x in items])
        history_box.markdown(
            f"<div style='background:#151515; border:1px solid #333; border-radius:6px; padding:8px;'>{html}</div>",
            unsafe_allow_html=True
        )


st.markdown(
    """
    <div style="color:#777; text-align:center; margin-top:30px; font-size:12px;">
        Monitoring Feed — BasuraNet
    </div>
    """,
    unsafe_allow_html=True
)
