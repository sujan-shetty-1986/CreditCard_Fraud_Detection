import numpy as np
import streamlit as st
import joblib

# Load pre-trained model
model = joblib.load('model_file.pkl')

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Define feature names
features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Create a grid layout for input fields
input_values = []
cols = st.columns(5)  # Create 5 columns for a grid layout

for i, feature in enumerate(features):
    col = cols[i % 5]  # Distribute inputs across columns
    value = col.number_input(f"{feature}", step=0.01)
    input_values.append(value)

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # Convert input values to numpy array
    features_array = np.array(input_values, dtype=np.float64).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features_array)
    # Display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
