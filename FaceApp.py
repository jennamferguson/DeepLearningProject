import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from keras_facenet import FaceNet
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Set page configuration
st.set_page_config(page_title="Are you part of the team?", layout="wide")


# Function to convert an image file to a base64 string for inline HTML (for logos)
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Update with the correct path to your logo image (if you have one)
jenna_path = "App_test_pics/Jenna.jpeg"
jenna_base64 = get_base64_image(jenna_path)

ramzi_path = "App_test_pics/Ramzi.jpg"
ramzi_base64 = get_base64_image(ramzi_path)

hrishi_path = "App_test_pics/Hrishi.jpg"
hrishi_base64 = get_base64_image(hrishi_path)

carson_path = "App_test_pics/Carson.Jpeg"
carson_base64 = get_base64_image(carson_path)

# Custom CSS for Jenna theme (light green, light purple, soft yellow)
custom_css = f"""
<style>
/* Set the page background to a light green */
.stApp {{
    background-color: #DFFFD6;
}}

/* Center text in the main container */
.main .block-container {{
    text-align: center;
}}

/* Title container to align logos and title text */
.title-container {{
    display: flex;
    align-items: center;
    justify-content: center;
}}
.title-container img {{
    height: 80px;
    margin: 0 10px;
}}
.title-container h1 {{
    color: #8A2BE2;  /* Purple for title text */
    margin: 0;
}}

/* Style paragraphs in the intro page to use black text */
.intro-text {{
    color: #000000;  /* Changed from #555555 to black */
    font-size: 18px;
    line-height: 1.5;
    background-color: #FFF9C4; /* soft yellow */
    padding: 20px;
    border-radius: 10px;
    display: inline-block;
}}

/* Override any other white text (like default Streamlit text) to be black */
body, .stApp, .block-container, p, span, div {{
    color: #000000 !important;
}}

/* Change the tab labels text to purple.
   Streamlit tabs use the Base Web library; this rule should target the tab element.
   (This selector may need adjustment if Streamlit updates its internals) */
[data-baseweb="tab"] > div[role="tab"] {{
    color: #8A2BE2 !important;
}}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Create tabs for navigation
tabs = st.tabs(["Introduction", "Jenna", "Ramzi", "Hrishi", "Carson"])

# INTRODUCTION TAB: Title with logos, description, and image uploader
with tabs[0]:
    st.markdown(
        f"""
    <div class="title-container">
        <img src="data:image/jpeg;base64,{carson_base64}" alt="Jenna Logo">
        <img src="data:image/jpeg;base64,{ramzi_base64}" alt="Jenna Logo">
        <img src="data:image/jpeg;base64,{hrishi_base64}" alt="Jenna Logo">
        <img src="data:image/jpeg;base64,{jenna_base64}" alt="Jenna Logo">
        <h1>Facial Verification for the Team</h1>
        <img src="data:image/jpeg;base64,{jenna_base64}" alt="Jenna Logo">
        <img src="data:image/jpeg;base64,{hrishi_base64}" alt="Jenna Logo">
        <img src="data:image/jpeg;base64,{ramzi_base64}" alt="Jenna Logo">
        <img src="data:image/jpeg;base64,{carson_base64}" alt="Jenna Logo">
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="intro-text">
        <b>Welcome!</b><br><br>
        This application is designed to determine if a person is part of our Deep Learning team. A potential member may upload a picture of their face,
        and from there we will identify whether or not they belong to the team.
        Our modeling approaches for this verification test have been fine-tuned using a pre-trained FaceNet model with an added binary classification layer.<br><br>
        <b>How it works:</b><br>
        - Upload your face image using the uploader below<br>
        - We will asses your face, and verify whether or not you belong to our team <br>
        - Navigate through each tab to see the more detail behind the our tests (and how close you might be to one of our members) <br><br>
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Image uploader on the intro page
    uploaded_file = st.file_uploader("Upload your face...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=500)
        # Store the uploaded image in session_state for use in other tabs
        st.session_state["uploaded_image"] = image
    else:
        st.info("Please upload an image to use for verification.")


# Helper function to process predictions if an image is available
def process_prediction(model):
    if "uploaded_image" not in st.session_state:
        st.warning(
            "No image uploaded yet. Please upload an image in the Introduction tab."
        )
        return None  # Return None if no image is available

    # Get and preprocess the uploaded image
    original_image = st.session_state["uploaded_image"]
    # Resize to match training (224x224 in your training code)
    resized_image = original_image.resize((224, 224))
    image_array = np.array(resized_image)
    # No normalization was done in training (based on your code snippet),
    # so we leave the values as-is. If needed, adjust accordingly.
    input_image = np.expand_dims(image_array, axis=0)

    # Get the prediction from the model
    prediction = model.predict(input_image)
    # Assuming prediction is a 2D array [[value]], extract the scalar
    pred_value = prediction[0][0]

    return original_image, pred_value


# Helper function to manually recreate the FaceNet-based architecture
def create_model():
    # Create the FaceNet model from keras_facenet
    embedder = FaceNet()
    base_model = embedder.model
    # Add your binary classification layer on top
    x = base_model.output
    classification_output = Dense(1, activation="sigmoid", name="classification_layer")(
        x
    )
    model = Model(inputs=base_model.input, outputs=classification_output)
    # Optionally freeze the base model if needed (adjust based on your training)
    for layer in base_model.layers:
        layer.trainable = True
    return model


# Jenna Page
with tabs[1]:
    st.markdown(
        "<h1 style='text-align: center;'>FaceNet trained for Jenna</h1>",
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_jenna_model():
        model = create_model()
        # Update with the actual path to your initial model weights file
        model.load_weights("Facenet_2.h5")
        return model

    FaceNet2 = load_jenna_model()

    # Get the resized image and prediction value
    result = process_prediction(FaceNet2)
    jenna_ref = Image.open("Jenna_ref.jpeg").resize((600, 600))

    # Only continue if an image was uploaded (result is not None)
    if result is not None:
        image, pred = result
        # image = image.resize((600, 700))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Uploaded Image")
        with col2:
            prediction_text = f"""
            <div style="text-align: center;">
                <h3>Prediction: {pred:.2f}</h3>
            """
            # Only do the conditional check if an image exists and a prediction was made
            if pred > 0.80:
                comment = (
                    "Since this is above our threshold set for correct identification, "
                    "this person is likely Jenna."
                )
            else:
                comment = (
                    "Since this is below our threshold for correct identification, "
                    "this person is likely not Jenna."
                )
            prediction_text += f"""There is a {pred*100:.2f}% 
            probability that this person is Jenna. {comment}
            """
            st.markdown(prediction_text, unsafe_allow_html=True)
        with col3:
            st.image(jenna_ref, caption="Reference Image")


# Ramzi page
with tabs[2]:
    st.markdown(
        "<h1 style='text-align: center;'>FaceNet trained for Ramzi</h1>",
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_ramzi_model():
        model = create_model()
        # Update with the actual path to your initial model weights file
        model.load_weights("Facenet_3.h5")
        return model

    FaceNet3 = load_ramzi_model()

    # Get the resized image and prediction value
    result = process_prediction(FaceNet3)
    ramzi_ref = Image.open("Ramzi_ref.jpeg").resize((600, 600))

    # Only continue if an image was uploaded (result is not None)
    if result is not None:
        image, pred = result
        # image = image.resize((600, 700))
        # Create two columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Uploaded Image")
        with col2:
            prediction_text = f"""
            <div style="text-align: center;">
                <h3>Prediction: {pred:.2f}</h3>
            """
            # Only do the conditional check if an image exists and a prediction was made
            if pred > 0.90:
                comment = (
                    "Since this is above our threshold set for correct identification, "
                    "this person is likely Ramzi."
                )
            else:
                comment = (
                    "Since this is below our threshold for correct identification, "
                    "this person is likely not Ramzi."
                )
            prediction_text += f"""There is a {pred*100:.2f}% 
            probability that this person is Ramzi. {comment}
            """
            st.markdown(prediction_text, unsafe_allow_html=True)
        with col3:
            st.image(ramzi_ref, caption="Reference Image")

# Hrishi page
with tabs[3]:
    st.markdown(
        "<h1 style='text-align: center;'>FaceNet trained for Hrishi</h1>",
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_hrishi_model():
        model = create_model()
        # Update with the actual path to your initial model weights file
        model.load_weights("Facenet_4.h5")
        return model

    FaceNet4 = load_hrishi_model()

    # Get the resized image and prediction value
    result = process_prediction(FaceNet4)
    hrishi_ref = Image.open("Hrishi_ref.jpeg").resize((600, 600))

    # Only continue if an image was uploaded (result is not None)
    if result is not None:
        image, pred = result
        # image = image.resize((600, 700))
        # Create two columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Uploaded Image")
        with col2:
            prediction_text = f"""
            <div style="text-align: center;">
                <h3>Prediction: {pred:.2f}</h3>
            """
            # Only do the conditional check if an image exists and a prediction was made
            if pred > 0.90:
                comment = (
                    "Since this is above our threshold set for correct identification, "
                    "this person is likely Hrishi."
                )
            else:
                comment = (
                    "Since this is below our threshold for correct identification, "
                    "this person is likely not Hrishi."
                )
            prediction_text += f"""There is a {pred*100:.2f}% 
            probability that this person is Hrishi. {comment}
            """
            st.markdown(prediction_text, unsafe_allow_html=True)
        with col3:
            st.image(hrishi_ref, caption="Reference Image")

with tabs[4]:
    st.markdown(
        "<h1 style='text-align: center;'>FaceNet trained for Carson</h1>",
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def load_carson_model():
        model = create_model()
        # Update with the actual path to your initial model weights file
        model.load_weights("Facenet_5.h5")
        return model

    FaceNet5 = load_carson_model()

    # Get the resized image and prediction value
    result = process_prediction(FaceNet5)
    carson_ref = Image.open("Carson_ref.jpeg").resize((600, 600))

    # Only continue if an image was uploaded (result is not None)
    if result is not None:
        image, pred = result
        # image = image.resize((600, 700))
        # Create two columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Uploaded Image")
        with col2:
            prediction_text = f"""
            <div style="text-align: center;">
                <h3>Prediction: {pred:.2f}</h3>
            """
            # Only do the conditional check if an image exists and a prediction was made
            if pred > 0.90:
                comment = (
                    "Since this is above our threshold set for correct identification, "
                    "this person is likely Carson."
                )
            else:
                comment = (
                    "Since this is below our threshold for correct identification, "
                    "this person is likely not Carson."
                )
            prediction_text += f"""There is a {pred*100:.2f}% 
            probability that this person is Carson. {comment}
            """
            st.markdown(prediction_text, unsafe_allow_html=True)
        with col3:
            st.image(carson_ref, caption="Reference Image")
