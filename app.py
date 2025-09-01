import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn import datasets

# Load trained model
model = joblib.load("svm_digits_model.pkl")

# Load dataset (for image reference)
digits = datasets.load_digits()

st.title("🖊️ Handwritten Digit Recognition with SVM")
st.write("Draw or select a digit image and the model will predict it.")

# User selects an index from dataset for demo
index = st.slider("Select a digit image (from dataset):", 0, len(digits.images)-1, 0)

# Normalize image to [0,1] range for Streamlit display
normalized_image = digits.images[index] / 16.0  # digits images are in range [0,16]

# Show image
st.image(normalized_image, caption=f"Digit: {digits.target[index]}", width=150)

# Flatten image for prediction
input_data = digits.data[index].reshape(1, -1)

# Predict
prediction = model.predict(input_data)[0]
st.success(f"✅ Predicted Digit: {prediction}")
