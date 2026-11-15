import joblib
import pandas as pd
import streamlit as st

st.title("Telco Customer Churn Predictor")
st.markdown("Enter customer details to predict churn probability.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/pipeline_v1.pkl")

pipeline = load_model()

# Input form
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Fixed defaults for remaining fields
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": monthly_charges * tenure,
    "Contract": contract,
    "InternetService": internet_service,
    "PaymentMethod": payment_method,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaperlessBilling": paperless_billing,
}])

if st.button("Predict Churn", type="primary"):
    probability = pipeline.predict_proba(input_data)[0][1]
    prediction = probability >= 0.5

    st.divider()

    if prediction:
        st.error(f"⚠️ High churn risk — {probability:.0%} probability")
    else:
        st.success(f"✅ Low churn risk — {probability:.0%} probability")

    st.metric("Churn Probability", f"{probability:.2%}")