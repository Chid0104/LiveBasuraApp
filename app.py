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


# =============================
#   STREAMLIT HEADER
# =============================
st.markdown("""
<h1 style='text-align:center; font-size:44px; margin-bottom:10px;'>
🗑️ BasuraNet — Live Trash Classifier
</h1>
<hr style='height:2px;border:none;background:#4CAF50;' />
""", unsafe_allow_html=True)


# =============================
#   VIDEO PROCESSOR
# =============================
class BasuraVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_pred = "Loading..."
        self.model = None
        self.model_loaded = False
        self.frame_count = 0
        
        # labels must match your TRAINED model
        self.labels = ["biodegradable", "recyclable", "residual"]

        # Confidence threshold for Unknown
        self.threshold = 0.65

        self.history = []
        self.load_model()


    # -------------------------
    #   MODEL LOADING
    # -------------------------
    def load_model(self):
        if not TF_AVAILABLE:
            self.last_pred = "TensorFlow missing"
            return
        
        model_path = "models/basuranet_final.h5"

        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
            self.last_pred = "Model Ready!"
        except:
            self.last_pred = "Model Load Error"


    # -------------------------
    #   PREDICTION
    # -------------------------
    def safe_predict(self, img):

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (224, 224))

        # Normalize correctly
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        preds = self.model.predict(img, verbose=0)[0]

        # Best class
        idx = int(np.argmax(preds))
        conf = float(preds[idx])

        # Classification logic
        if conf < self.threshold:
            return "Unknown", conf
        else:
            return self.labels[idx], conf


    # -------------------------
    #   RECEIVE FRAME
    # -------------------------
    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.model_loaded and self.frame_count % 10 == 0:
            try:
                label, conf = self.safe_predict(img)
                self.last_pred = f"{label} ({conf:.2f})"

                # update history
                if label != "Unknown":
                    self.history.append(self.last_pred)
                    if len(self.history) > 8:
                        self.history.pop(0)

            except:
                self.last_pred = "Prediction Error"


        # Overlay text
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



# =============================
#   STREAMLIT UI
# =============================
col1, col2 = st.columns([4,2])

with col1:
    webrtc_streamer(
        key="basuranet",
        video_processor_factory=BasuraVideoProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        },
    )

with col2:
    st.subheader("📝 Prediction History")
    placeholder = st.empty()


# Update history output
def update_history(processor):
    if processor and hasattr(processor, "history"):
        placeholder.write("\n".join(processor.history))


st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Hold trash in front of the camera and wait for stable detection.")
