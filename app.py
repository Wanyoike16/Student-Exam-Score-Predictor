import streamlit as st
import numpy as np
import joblib
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load model
model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

# User inputs
study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 2.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.9)
part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

# Prediction
if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)
    predicted_score = float(prediction[0])
    predicted_score = max(0, min(100, predicted_score))  # Clamp score

    st.success(f"Predicted Exam Score: {predicted_score:.2f}")

    # --- SHAP Explanation ---
    st.subheader("Feature Contribution (SHAP)")
     # Define feature names
    feature_names = ["Study Hours", "Attendance", "Mental Health", "Sleep Hours", "Part-Time Job"]
    # Create SHAP explainer
    explainer = shap.LinearExplainer(model, np.array([[0]*5]))  # dummy background for explanation

    shap_values = explainer(input_data)

    # Plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                      base_values=explainer.expected_value, 
                                      data=input_data[0], 
                                      feature_names=feature_names), max_display=5, show=False)
    st.pyplot(fig)

