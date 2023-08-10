from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

model = load_model("keras_model.h5", compile=False)

class_names = ["0", "1"]  # Make sure the class names match the order in your model

def classify_image(img):
    # Convert to RGB mode to ensure 3 channels
    img = img.convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the model
    confidence_score = model.predict(data)
    predicted_class = np.argmax(confidence_score)

    return predicted_class

def main():
    st.title("Image Classification")
    st.header("Pneumonia X-Ray Classifications")
    st.text("Upload a Pneumonia X-Ray for classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        predicted_class = classify_image(img)

        st.write("Uploaded Image:")
        st.image(img, use_column_width=True)

        if predicted_class == 1:
            st.success("Pneumonia found!")
        else:
            st.error("No Pneumonia Found.")
        st.write(f"Classified as: {class_names[predicted_class]}")

if __name__ == "__main__":
    main()
