import streamlit as st
from functions_face_emotion import display_model


st.title('Emotion Recognition with CNN')
st.markdown(
    """
    Welcome to the Emotion Recognition application powered by a Convolutional Neural Network (CNN).
    
    
    This application is designed to analyze facial expressions in real-time using your webcam.
    Click the "Start Webcam" button below to begin capturing images and predicting emotions.
    """
)

display_model()

