from tensorflow.keras.layers import LSTM
from tensorflow.keras.saving import register_keras_serializable
import streamlit as st
import tensorflow as tf
import numpy as np
import os
os.environ['TF_GRAPPLER_DISABLE'] = '1'

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'toxicity.h5',
        custom_objects={'LSTM': tf.keras.layers.LSTM}
    )


model = load_model()

# Set app title
st.title('Toxicity Detection App')

# Input text box
user_input = st.text_area("Enter text to analyze for toxicity:", "")

# Prediction button
if st.button('Analyze'):
    if user_input:
        # Preprocess input (modify according to your model's requirements)
        # Example: Convert text to a Tensor (adjust preprocessing steps)
        input_data = tf.convert_to_tensor([user_input])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result (modify based on your model's output format)
        toxicity_score = prediction[0][0]  # Adjust index if needed
        st.write(f"Toxicity Score: {toxicity_score:.4f}")
        
        # Optional: Add a threshold-based message
        threshold = 0.5
        if toxicity_score > threshold:
            st.error("This text is likely toxic!")
        else:
            st.success("This text appears safe.")
    else:
        st.warning("Please enter some text to analyze!")
