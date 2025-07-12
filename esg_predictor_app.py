# esg_predictor_app.py

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset and train the models (this can be saved and reused for faster loading)

@st.cache_data
def load_and_train_models():
    df = pd.read_csv("SP 500 ESG Risk Ratings.csv")
    
    # Clean missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    df = df.dropna()

    target = 'Total ESG Risk score'
    features = num_cols.drop(target)
    X = df[features]
    y = df[target]

    # Scale for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    rf = RandomForestRegressor().fit(X, y)
    xgbr = xgb.XGBRegressor(objective='reg:squarederror').fit(X, y)
    svr = SVR().fit(X_scaled, y)

    return rf, xgbr, svr, features, scaler

# Load models and columns
rf_model, xgb_model, svm_model, feature_cols, scaler = load_and_train_models()

st.title("🌱 ESG Risk Score Predictor")
st.markdown("Enter company data to predict its **Total ESG Risk Score** using ML models.")

# Create input form dynamically
input_data = {}
for col in feature_cols:
    val = st.number_input(f"{col}", min_value=0.0, step=0.1)
    input_data[col] = val

input_df = pd.DataFrame([input_data])

# Predict on input
if st.button("🔮 Predict ESG Score"):
    rf_pred = rf_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]
    svm_pred = svm_model.predict(scaler.transform(input_df))[0]

    st.success(f"🌲 Random Forest Prediction: **{rf_pred:.2f}**")
    st.success(f"⚡ XGBoost Prediction: **{xgb_pred:.2f}**")
    st.success(f"📈 SVM Prediction: **{svm_pred:.2f}**")

    st.bar_chart(pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "SVM"],
        "Predicted ESG Score": [rf_pred, xgb_pred, svm_pred]
    }).set_index("Model"))
