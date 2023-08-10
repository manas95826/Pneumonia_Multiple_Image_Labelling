from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import streamlit as st

model = load_model("keras_model.h5", compile=False)

class_names = ["0", "1"]  # Make sure the class names match the order in your model

def classify_image(img):
    # Check if the image is in RGB format
    if img.mode != "L":
        return "Invalid"

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = img.convert('RGB')  # Convert to RGB mode to ensure 3 channels
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    confidence_score = model.predict(data)
    predicted_class = np.argmax(confidence_score)

    return predicted_class

def main():
    st.title("Image Classification")
    st.header("Pneumonia X-Ray Classification")
    st.text("Upload a Pneumonia X-Ray for classification")

    folder = st.text_input("Enter folder path containing images:", type="default")
    
    if folder and os.path.exists(folder):
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg'))]
        
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            img = Image.open(image_path)
            predicted_class = classify_image(img)
            
            st.write(f"Image: {image_file}")
            if predicted_class == 1:
                st.success("Pneumonia found!")
            else:
                st.error("No Pneumonia Found.")
            
            st.image(img, caption=f"Classified as: {class_names[predicted_class]}", use_column_width=True)
    else:
        st.warning("Please provide a valid folder path")

if __name__ == "__main__":
    main()