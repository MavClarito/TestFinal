import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Cache the model to avoid reloading it on every run
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Cifar10_Model.h5')
    return model

model = load_model()

# Streamlit UI
st.write("# CIFAR10 Detection System")
file = st.file_uploader("Insert Image", type=["jpg", "png"])

# Image preprocessing and prediction
def import_and_predict(image_data, model):
    try:
        # Resize the image to (32, 32) and ensure RGB format
        size = (32, 32)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS).convert('RGB')

        # Normalize the image to [0, 1]
        img_array = np.asarray(image) / 255.0

        # Add a batch dimension
        img_reshape = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Main logic
if file is None:
    st.text("Please upload an image file.")
else:
    try:
        # Display uploaded image
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Predict the class
        prediction = import_and_predict(image, model)

        if prediction is not None:
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            pred_class = class_names[np.argmax(prediction)]
            st.success(f"This image is classified as: **{pred_class}**")
        else:
            st.error("Failed to classify the image.")
    except Exception as e:
        st.error(f"Error: {e}")
