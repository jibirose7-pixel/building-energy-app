
import streamlit as st
import numpy as np
import mlflow
import mlflow.tensorflow
import joblib
import tensorflow as tf

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Building Energy Load Prediction", layout="centered")
st.title("Building Energy Load Prediction")
st.write("Neural Network model loaded from MLflow")

# ----------------------------
# Load scaler + PCA
# ----------------------------
@st.cache_resource
def load_model_and_preprocessors():
    model = tf.keras.models.load_model(
        'energy_model.keras')

    # Load scaler and PCA
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")


    return model, scaler, pca

model, scaler, pca = load_model_and_preprocessors()

# ----------------------------
# Sidebar for user input
# ----------------------------
st.sidebar.header("Building Parameters")

rc = st.sidebar.slider("Relative Compactness", 0.5, 1.0, 0.75)
sa = st.sidebar.slider("Surface Area", 500.0, 900.0, 700.0)
wa = st.sidebar.slider("Wall Area", 200.0, 400.0, 300.0)
ra = st.sidebar.slider("Roof Area", 100.0, 300.0, 200.0)
oh = st.sidebar.slider("Overall Height", 3.5, 7.0, 5.0)
ori = st.sidebar.selectbox("Orientation", [2, 3, 4, 5])
ga = st.sidebar.slider("Glazing Area", 0.0, 0.4, 0.2)
gad = st.sidebar.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

input_data = np.array([[rc, sa, wa, ra, oh, ori, ga, gad]])

# ----------------------------
# Preprocessing
# ----------------------------
input_scaled = scaler.transform(input_data)
input_pca = pca.transform(input_scaled)

# ----------------------------
# Prediction button
# ----------------------------
if st.button("Predict Energy Load"):
    prediction = model.predict(input_pca)
    st.success(f"Heating Load: {prediction[0, 0]:.2f}")
    st.success(f"Cooling Load: {prediction[0, 1]:.2f}")


