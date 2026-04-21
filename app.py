
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load the trained model, scaler, and encoders
# Ensure these files are in the same directory as your app.py on Streamlit Cloud
model = joblib.load('XGBoost_best_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoder.pkl')

# Define the preprocessing function
def preprocess_input(input_df, scaler, encoders):
    processed_df = input_df.copy()

    # Apply Label Encoding for 'Product_importance' and 'Gender'
    for col in ['Product_importance', 'Gender']:
        if col in encoders:
            le = encoders[col]
            processed_df[col] = le.transform(processed_df[col])
        else:
            st.error(f"LabelEncoder for {col} not found.")
            return None

    # One-Hot Encode nominal categorical columns
    warehouse_block_dummies = pd.get_dummies(processed_df['Warehouse_block'], prefix='Warehouse_block', drop_first=True)
    mode_of_shipment_dummies = pd.get_dummies(processed_df['Mode_of_Shipment'], prefix='Mode_of_Shipment', drop_first=True)

    processed_df = pd.concat([processed_df.drop(columns=['Warehouse_block', 'Mode_of_Shipment']), 
                              warehouse_block_dummies, 
                              mode_of_shipment_dummies], axis=1)

    # Ensure all boolean columns are converted to int (0/1)
    for col in processed_df.select_dtypes(include='bool').columns:
        processed_df[col] = processed_df[col].astype(int)

    # Feature Engineering (same as in training)
    processed_df['Cost_to_Weight_ratio'] = processed_df['Cost_of_the_Product'] / processed_df['Weight_in_gms']
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_df['Cost_to_Weight_ratio'].fillna(processed_df['Cost_to_Weight_ratio'].median(), inplace=True)

    processed_df['Cost*Weight'] = processed_df['Cost_of_the_Product'] * processed_df['Weight_in_gms']
    processed_df['Discount_Ratio'] = processed_df['Discount_offered'] / processed_df['Cost_of_the_Product']
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_df.fillna(0, inplace=True) # Fill any remaining NaNs after new features

    processed_df['CareCalls_to_Purchases'] = processed_df['Customer_care_calls'] / (processed_df['Prior_purchases'] + 1)
    processed_df['CostWeight_Discount_Interaction'] = processed_df['Cost_to_Weight_ratio'] * (processed_df['Discount_offered'] + 1)
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_df.fillna(processed_df.median(numeric_only=True), inplace=True)

    # Define numerical columns for scaling
    num_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
                'Prior_purchases', 'Discount_offered', 'Weight_in_gms',
                'Cost_to_Weight_ratio', 'Cost*Weight', 'Discount_Ratio',
                'CareCalls_to_Purchases', 'CostWeight_Discount_Interaction']

    # Apply StandardScaler to numerical columns
    processed_df[num_cols] = scaler.transform(processed_df[num_cols])

    # Ensure all columns from training are present and in the correct order
    # (This is a crucial step for consistent predictions)
    # Get columns from the original training data (assuming X_train has all features in correct order)
    # You would need to save X_train.columns to a file or hardcode them if not available
    # For this example, let's assume `model.feature_names_in_` is available or reconstruct from notebook state

    # Manually reconstruct the columns based on the notebook's X_train (excluding 'ID' for model input if not used)
    # Assuming 'ID' was dropped before training the actual model input features
    # The columns from the notebook state (X_train) are:
    training_columns = ['ID', 'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
                        'Prior_purchases', 'Product_importance', 'Gender', 'Discount_offered',
                        'Weight_in_gms', 'Warehouse_block_B', 'Warehouse_block_C',
                        'Warehouse_block_D', 'Warehouse_block_F', 'Mode_of_Shipment_Road',
                        'Mode_of_Shipment_Ship', 'Cost_to_Weight_ratio', 'Cost*Weight',
                        'Discount_Ratio', 'CareCalls_to_Purchases', 'CostWeight_Discount_Interaction']

    # Drop 'ID' if it's not used by the model for prediction (which is typical)
    if 'ID' in processed_df.columns and 'ID' not in model.feature_names_in_:
        processed_df = processed_df.drop(columns=['ID'])
        training_columns.remove('ID') # Adjust training_columns if ID is dropped

    # Reorder and add missing columns with default value 0
    final_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else training_columns

    for col in final_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0 # Add missing one-hot encoded columns as 0
    processed_df = processed_df[final_columns]

    return processed_df

# Streamlit UI
st.set_page_config(page_title="ShipmentSure Predictor", layout="wide")
st.title("📦 ShipmentSure: On-Time Delivery Predictor")
st.markdown("Enter shipment details to predict if it will be delivered on time.")

with st.form("prediction_form"):
    st.header("Shipment Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        warehouse_block = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
        mode_of_shipment = st.selectbox("Mode of Shipment", ['Flight', 'Road', 'Ship'])
        customer_care_calls = st.slider("Customer Care Calls", 2, 7, 4)
        customer_rating = st.slider("Customer Rating (1-5)", 1, 5, 3)

    with col2:
        cost_of_the_product = st.number_input("Cost of the Product (in USD)", 90, 320, 200)
        prior_purchases = st.slider("Prior Purchases", 2, 10, 3)
        product_importance = st.selectbox("Product Importance", ['low', 'medium', 'high'])
        gender = st.selectbox("Gender", ['M', 'F'])

    with col3:
        discount_offered = st.number_input("Discount Offered (in %)", 1, 65, 10)
        weight_in_gms = st.number_input("Weight in gms", 1000, 8000, 3000)

    submit_button = st.form_submit_button("Predict Delivery Status")

    if submit_button:
        # Create a DataFrame from input
        input_data = pd.DataFrame({
            'ID': [0], # Placeholder ID
            'Warehouse_block': [warehouse_block],
            'Mode_of_Shipment': [mode_of_shipment],
            'Customer_care_calls': [customer_care_calls],
            'Customer_rating': [customer_rating],
            'Cost_of_the_Product': [cost_of_the_product],
            'Prior_purchases': [prior_purchases],
            'Product_importance': [product_importance],
            'Gender': [gender],
            'Discount_offered': [discount_offered],
            'Weight_in_gms': [weight_in_gms]
        })

        # Preprocess the input data
        processed_input = preprocess_input(input_data, scaler, encoders)

        if processed_input is not None:
            # Make prediction
            prediction = model.predict(processed_input)
            prediction_proba = model.predict_proba(processed_input)[:, 1]

            st.subheader("Prediction Results:")
            if prediction[0] == 1:
                st.success(f"The shipment is predicted to **reach on time!** (Probability: {prediction_proba[0]:.2f})")
            else:
                st.error(f"The shipment is predicted to **NOT reach on time.** (Probability: {prediction_proba[0]:.2f})")

            st.write("--- ")
            st.info("Disclaimer: This prediction is based on the trained model and provided input. It should be used for informational purposes only.")


