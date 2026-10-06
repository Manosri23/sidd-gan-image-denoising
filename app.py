import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="SIDD GAN Image Denoising",
    page_icon="🖼️",
    layout="centered"
)

# Title
st.title("🖼️ SIDD GAN Image Denoising")
st.write(
    "Remove unwanted noise from real-scene images using a "
    "Deep Learning GAN model."
)

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("generator.keras")
    return model

try:
    generator = load_model()
    st.success("GAN model loaded successfully!")
except Exception as e:
    st.error("Model could not be loaded.")
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a noisy image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Noisy Image")
    st.image(image, caption="Input Image", use_container_width=True)

    # Preprocessing
    image_array = np.array(image).astype("float32") / 255.0

    # Resize to model input size
    image_array = tf.image.resize(
        image_array,
        [128, 128]
    )

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Denoising
    if st.button("✨ Remove Noise"):

        with st.spinner("Processing image..."):

            denoised_image = generator.predict(
                image_array,
                verbose=0
            )

        # Convert output to displayable image
        denoised_image = np.clip(
            denoised_image[0],
            0,
            1
        )

        denoised_image = (denoised_image * 255).astype(
            np.uint8
        )

        st.subheader("Denoised Image")
        st.image(
            denoised_image,
            caption="GAN Denoised Image",
            use_container_width=True
        )

        st.success("Noise removal completed successfully! ✅")
