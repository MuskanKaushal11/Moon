import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import io
from PIL import Image
from datetime import datetime
import pandas as pd
from tensorflow.keras.utils import register_keras_serializable

st.set_page_config(
    page_title="🌖 MOONARC: Real-Time Lunar Phase Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# 1) Register custom preprocessing so Keras can deserialize the Lambda
# -------------------------------------------------------------------
@register_keras_serializable()
def resnet_preprocess(img):
    from tensorflow.keras.applications.resnet import preprocess_input
    return preprocess_input(img)

# -------------------------------------------------------------------
# 2) Load model (with the correct custom_object name)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    tf.keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(
        'model.keras',
        compile=False,
        custom_objects={'resnet_preprocess': resnet_preprocess}
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------------------------------------------
# 3) CLAHE enhancement
# -------------------------------------------------------------------
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

# -------------------------------------------------------------------
# 4) Detect and crop the moon (no contour draw)
# -------------------------------------------------------------------
def detect_and_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (x,y),r = cv2.minEnclosingCircle(c)
    x,y,r = int(x), int(y), int(r)
    y1, y2 = max(y-r,0), min(y+r,image.shape[0])
    x1, x2 = max(x-r,0), min(x+r,image.shape[1])
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None

# -------------------------------------------------------------------
# 5) Full preprocessing pipeline
# -------------------------------------------------------------------
def preprocess_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, False
    crop = detect_and_crop(img)
    if crop is None:
        # no moon found: return original
        return img, False
    return apply_clahe(crop), True

# -------------------------------------------------------------------
# 6) Prediction helper
# -------------------------------------------------------------------
def predict_phase(img, model):
    img = cv2.resize(img, (224,224))
    x = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    p = model.predict(x)
    names = [
      'first quarter','full moon','new moon','no moon',
      'third quarter','waning crescent','waning gibbous',
      'waxing crescent','waxing gibbous'
    ]
    idx = np.argmax(p[0])
    return names[idx], float(100 * tf.reduce_max(tf.nn.softmax(p[0])))

# -------------------------------------------------------------------
# 7) Streamlit UI
# -------------------------------------------------------------------
def main():
    st.title("🌖 MOONARC")
    st.markdown("**Real-Time Lunar Phase Detection**", unsafe_allow_html=True)
    model = load_model()

    with st.sidebar:
        st.header("Input Method")
        method = st.radio("", ("Upload Image", "Camera"))
        img_bytes = None
        if method == "Upload Image":
            f = st.file_uploader("Choose JPG/PNG", type=["jpg","jpeg","png"])
            if f:
                img_bytes = f.read()
        else:
            c = st.camera_input("Take a photo")
            if c:
                img_bytes = c.read()

        st.markdown("---")
        st.header("About")
        st.write("MoonArc detects lunar phases using a CNN backend and Flutter frontend.")

    if img_bytes:
        proc_img, found = preprocess_image(img_bytes)
        orig = Image.open(io.BytesIO(img_bytes))

        col1, col2 = st.columns(2)
        col1.image(orig, caption="Original Image", use_container_width=True)
        if not found:
            col2.image(orig, caption="No Moon Detected", use_container_width=True)
        else:
            col2.image(proc_img[:,:,::-1], caption="Processed Moon Crop", use_container_width=True)

        if found:
            phase, conf = predict_phase(proc_img, model)
            st.metric("Prediction", f"{phase}", delta=f"{conf:.1f}%")
        else:
            st.error("Could not detect a moon. Try another image.")

        # Feedback section
        st.markdown("---")
        st.header("Feedback")
        if found:
            ans = st.radio("Was this correct?", ("Yes","No"))
            if ans == "No":
                correct = st.selectbox("Select correct phase", [
                    'first quarter','full moon','new moon','no moon',
                    'third quarter','waning crescent','waning gibbous',
                    'waxing crescent','waxing gibbous'
                ])
                if st.button("Submit Feedback"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("Feedbacks/images", exist_ok=True)
                    path = f"Feedbacks/images/{ts}.jpg"
                    with open(path,"wb") as f:
                        f.write(img_bytes)
                    df = pd.DataFrame([{
                        "timestamp": ts,
                        "path": path,
                        "pred": phase,
                        "correct": correct
                    }])
                    os.makedirs("Feedbacks", exist_ok=True)
                    df.to_csv(
                        "Feedbacks/feedback.csv",
                        mode='a', index=False,
                        header=not os.path.exists("Feedbacks/feedback.csv")
                    )
                    st.success("Thanks! Your feedback helps improve MoonArc.")

if __name__ == "__main__":
    main()
