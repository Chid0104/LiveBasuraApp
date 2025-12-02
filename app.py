import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np
from PIL import Image
import time

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

class BasuraVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_pred = "No prediction"
        self.model_loaded = False

    def load_model(self):
        if TF_AVAILABLE:
            self.model = tf.keras.models.load_model("models/basura_model")
            self.model_loaded = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Inference every X frames
        if not self.model_loaded:
            try:
                self.load_model()
            except:
                pass

        if self.model_loaded:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((224,224))
            arr = np.expand_dims(np.array(pil)/255.0, axis=0)

            pred = self.model.predict(arr)[0]
            idx = np.argmax(pred)
            labels = ["biodegradable", "recyclable", "residual"]

            self.last_pred = labels[idx]

        # Overlay text
        cv2.putText(
            img, self.last_pred.upper(), 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 0), 2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("BasuraNet — Live Waste Classifier")

webrtc_streamer(
    key="basuranet",
    video_processor_factory=BasuraVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
