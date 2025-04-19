import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Sidebar menu
st.sidebar.title("📂 Menu")
selection = st.sidebar.radio("Go to", ["Image Classifier", "About"])

# Load the model only once
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('flower_model_trained.hdf5', compile=False)

# Function to preprocess image
def preprocess_image(img, target_size=(176, 176)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # remove alpha channel
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Page 1: Image Classifier
if selection == "Image Classifier":
    st.title("Image Classifier")
    st.write("Upload a image and let the model predict.")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Classifying...")

        processed_image = preprocess_image(image)
        model = load_trained_model()

        try:
            prediction = model.predict(processed_image)

            # Replace these with your actual class labels
            class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            st.success(f"Prediction: **{predicted_class}** with {confidence * 100:.2f}% confidence")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif selection == "Classify":
    st.title("🌸 Flower Image Classification")
    uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        # Preprocess image
        processed = image.resize((224, 224))
        processed = np.array(processed) / 255.0  # normalize
        processed = np.expand_dims(processed, axis=0)  # shape (1, 224, 224, 3)

        # Predict
        prediction = model.predict(processed)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"🌼 **Predicted Flower:** {predicted_class}")
        st.info(f"🔍 **Confidence:** {confidence * 100:.2f}%")


# Page 2: About
elif selection == "About":
    st.title("📘 About This Project")
    st.markdown("---")

    st.markdown("### Project Title: **Image Classification using Transfer Learning**")

    st.markdown("#### 🧠 Objective:")
    st.write("""
    This project is designed to classify images of flowers into five categories using a deep learning model 
    built with TensorFlow and deployed using Streamlit.
    """)

    st.markdown("#### 🔧 Technologies Used:")
    st.markdown("""
    - **Python 3.12**
    - **TensorFlow 2.16.1**
    - **Streamlit**
    - **Pillow**
    - **NumPy**
    """)

    st.markdown("#### 👩‍💻 Team Members:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- 🟣 **Ishwari Kshirsagar**")
        st.markdown("- 🔵 **Monika Yawale**")
    with col2:
        st.markdown("- 🟢 **Janhavi Baraskar**")
        st.markdown("- 🟡 **Vaishnavi Baraskar**")

    st.markdown("#### 📁 Dataset & Model Info:")
    st.write("""
    The model is trained on a flower dataset consisting of **Daisy**, **Dandelion**, **Rose**, **Sunflower**, and **Tulip** images.
    We used **Transfer Learning** with a pre-trained CNN architecture to achieve high accuracy with limited data.
    """)

    st.markdown("#### 🚀 Features:")
    st.markdown("""
    - Upload flower images for instant classification
    - Confidence score shown for predictions
    - Clean and interactive UI built with Streamlit
    """)

    st.markdown("#### 🗃️ Source Code:")
    st.markdown("[🌐 View on GitHub](https://github.com/yourusername/Image-Classification-Streamlit-TensorFlow)")

    st.markdown("---")
    st.markdown("Made with ❤️ by **Team Innovators**")



