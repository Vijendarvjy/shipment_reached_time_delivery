import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# LOAD ARTIFACTS (SAFE + CACHED)
# -------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('XGBoost_best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoders = joblib.load('encoder.pkl')
        return model, scaler, encoders
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        st.stop()

model, scaler, encoders = load_artifacts()

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess_input(input_df):
    df = input_df.copy()

    # -------------------------------
    # LABEL ENCODING (SAFE)
    # -------------------------------
    for col in ['Product_importance', 'Gender']:
        if col in encoders:
            le = encoders[col]

            if df[col].iloc[0] not in le.classes_:
                st.error(f"Unknown category '{df[col].iloc[0]}' in {col}")
                st.stop()

            df[col] = le.transform(df[col])
        else:
            st.error(f"Encoder missing for {col}")
            st.stop()

    # -------------------------------
    # ONE-HOT ENCODING
    # -------------------------------
    df = pd.get_dummies(df, columns=['Warehouse_block', 'Mode_of_Shipment'], drop_first=True)

    # -------------------------------
    # SCALE ONLY BASE NUMERICAL FEATURES
    # -------------------------------
    try:
        scaler_cols = scaler.feature_names_in_
    except:
        scaler_cols = [
            'Customer_care_calls',
            'Customer_rating',
            'Cost_of_the_Product',
            'Prior_purchases',
            'Discount_offered',
            'Weight_in_gms'
        ]

    for col in scaler_cols:
        if col not in df.columns:
            df[col] = 0

    df[scaler_cols] = scaler.transform(df[scaler_cols])

    # -------------------------------
    # FEATURE ENGINEERING (AFTER SCALING)
    # -------------------------------
    df['Cost_to_Weight_ratio'] = df['Cost_of_the_Product'] / (df['Weight_in_gms'] + 1)
    df['Cost*Weight'] = df['Cost_of_the_Product'] * df['Weight_in_gms']
    df['Discount_Ratio'] = df['Discount_offered'] / (df['Cost_of_the_Product'] + 1)
    df['CareCalls_to_Purchases'] = df['Customer_care_calls'] / (df['Prior_purchases'] + 1)
    df['CostWeight_Discount_Interaction'] = df['Cost_to_Weight_ratio'] * (df['Discount_offered'] + 1)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # -------------------------------
    # FINAL COLUMN ALIGNMENT
    # -------------------------------
    try:
        final_cols = model.feature_names_in_
    except:
        final_cols = df.columns

    for col in final_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[final_cols]

    return df


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Shipment Predictor", layout="wide")
st.title("📦 Shipment On-Time Delivery Predictor")

st.markdown("Enter shipment details below:")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        warehouse_block = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
        mode = st.selectbox("Mode of Shipment", ['Flight', 'Road', 'Ship'])
        care_calls = st.slider("Customer Care Calls", 2, 7, 4)
        rating = st.slider("Customer Rating", 1, 5, 3)

    with col2:
        cost = st.number_input("Cost of Product", 90, 320, 200)
        purchases = st.slider("Prior Purchases", 2, 10, 3)
        importance = st.selectbox("Product Importance", ['low', 'medium', 'high'])
        gender = st.selectbox("Gender", ['M', 'F'])

    with col3:
        discount = st.number_input("Discount Offered (%)", 1, 65, 10)
        weight = st.number_input("Weight (gms)", 1000, 8000, 3000)

    submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame({
            'ID': [0],
            'Warehouse_block': [warehouse_block],
            'Mode_of_Shipment': [mode],
            'Customer_care_calls': [care_calls],
            'Customer_rating': [rating],
            'Cost_of_the_Product': [cost],
            'Prior_purchases': [purchases],
            'Product_importance': [importance],
            'Gender': [gender],
            'Discount_offered': [discount],
            'Weight_in_gms': [weight]
        })

        processed = preprocess_input(input_df)

        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(f"✅ On-Time Delivery (Confidence: {probability:.2f})")
        else:
            st.error(f"❌ Delayed Delivery (Confidence: {probability:.2f})")

        st.progress(float(probability))

        if probability > 0.8:
            st.write("🟢 High Confidence")
        elif probability > 0.6:
            st.write("🟡 Medium Confidence")
        else:
            st.write("🔴 Low Confidence")

        st.info("⚠️ This prediction is based on ML model and may not be 100% accurate.")
