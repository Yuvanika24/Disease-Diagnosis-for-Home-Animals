import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('widen.h5')

# Define your class names
class_names = ['BUMPS', 'HAIR LOSS', 'HOT SPOTS', 'RASHES', 'SORES']

# Function to preprocess the image
def preprocess_image(img):
    # Resize the image to match the input size of the model
    img_resized = cv2.resize(img, (128, 128))
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    return img_resized

# Define the Streamlit app
def main():
    st.title('Skin Disease Classifier')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Preprocess the image
        processed_img = preprocess_image(img)

        # Make prediction
        prediction = model.predict(processed_img)

        # Get class name
        class_idx = np.argmax(prediction)
        class_name = class_names[class_idx]

        # Resize the image for display
        resized_img = cv2.resize(img, (256, 256))

        # Display the resized image
        st.image(resized_img, caption='Uploaded Image', use_column_width=True)

        # Display the prediction
        st.write('Prediction:', f'<span style="font-size: 20px; font-weight: bold; color: blue;">{class_name}</span>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

