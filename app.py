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

class BasuraVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_pred = "Starting..."
        self.model_loaded = False
        self.model = None
        self.frame_count = 0
        self.labels = ["biodegradable", "recyclable", "residual"]
        
        # Try to load model immediately
        self.load_model()

    def load_model(self):
        """Load the .h5 model"""
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
        
        # Run inference every 20 frames for performance
        self.frame_count += 1
        
        if self.frame_count % 20 == 0:
            if not self.model_loaded:
                try:
                    self.load_model()
                except:
                    pass
            
            if self.model_loaded and self.model is not None:
                try:
                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize for model input
                    input_img = cv2.resize(rgb, (224, 224))
                    
                    # Normalize and add batch dimension
                    input_array = np.expand_dims(input_img / 255.0, axis=0)
                    
                    # Make prediction
                    predictions = self.model.predict(input_array, verbose=0)[0]
                    class_idx = np.argmax(predictions)
                    confidence = predictions[class_idx]
                    
                    # Get label
                    if class_idx < len(self.labels):
                        label = self.labels[class_idx]
                    else:
                        label = f"Class {class_idx}"
                    
                    # Update prediction text
                    if confidence > 0.5:
                        self.last_pred = f"{label} ({confidence:.2f})"
                    else:
                        self.last_pred = "Uncertain"
                        
                except Exception as e:
                    self.last_pred = "Pred error"
                    # print(f"Prediction error: {e}")

        # Overlay prediction text
        cv2.putText(
            img, 
            self.last_pred, 
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        # Add frame counter (optional)
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

st.title("🗑️ BasuraNet — Live Waste Classifier")

# Display model status
st.write("### Model Status")
if TF_AVAILABLE:
    st.success("✅ TensorFlow is available")
    
    # Check if .h5 model exists
    model_path = "models/basuranet_final.h5"
    if os.path.exists(model_path):
        st.success(f"✅ Model found: {model_path}")
        st.info(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    else:
        st.error(f"❌ Model not found at: {model_path}")
        st.write("Please ensure your model is at: `models/basuranet_final.h5`")
else:
    st.error("❌ TensorFlow is not available")
    st.code("pip install tensorflow>=2.12.0")

# Display labels
st.write("### Classification Labels")
st.write("1. **Biodegradable** ♻️ - Food waste, leaves, paper")
st.write("2. **Recyclable** 🔄 - Plastic bottles, cans, cardboard")
st.write("3. **Residual** 🗑️ - Diapers, ceramics, mixed waste")

st.info("💡 **Tip:** Hold waste item in the camera view and wait for prediction")

# Start the camera
webrtc_streamer(
    key="basuranet",
    video_processor_factory=BasuraVideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False
    },
)

# Footer with instructions
st.write("---")
st.write("**Instructions:**")
st.write("1. Click 'START' to begin camera")
st.write("2. Hold waste item in front of camera")
st.write("3. Wait for prediction to appear in green text")
st.write("4. Prediction updates every ~0.7 seconds")
