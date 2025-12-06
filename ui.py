import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Title and CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Credit Card Fraud Detection System")
st.write("Built with XGBoost, SMOTE, and Streamlit")

# Load Resources
@st.cache_resource
def load_resources():
    model = joblib.load('fraud_model.pkl')
    # We load the sample test data we saved earlier to simulate transactions
    X_test = pd.read_csv("test_data_sample.csv")
    y_test = pd.read_csv("test_labels_sample.csv")
    return model, X_test, y_test

try:
    model, X_test, y_test = load_resources()
    st.success("Model loaded successfully!")
except:
    st.error("Please run the training script first to generate 'fraud_model.pkl' and data samples.")
    st.stop()

# Sidebar
st.sidebar.header("User Input / Simulation")
st.sidebar.info("Since real CC data has 28 anonymized PCA features, manual entry is difficult. Click below to simulate a transaction from the test dataset.")

if st.sidebar.button("🎲 Simulate Random Transaction"):
    # Select a random row from the test set
    random_index = np.random.randint(0, len(X_test))
    transaction = X_test.iloc[random_index]
    true_label = y_test.iloc[random_index].values[0]
    
    # Display Transaction Details
    st.subheader("Transaction Details")
    col1, col2, col3 = st.columns(3)
    col1.metric("Time (Scaled)", round(transaction['Time'], 2))
    col2.metric("Amount (Scaled)", round(transaction['Amount'], 2))
    col3.metric("Actual Label", "Fraud 🚨" if true_label == 1 else "Legit ✅")

    # Make Prediction
    transaction_data = transaction.values.reshape(1, -1)
    prediction = model.predict(transaction_data)[0]
    probability = model.predict_proba(transaction_data)[0][1]

    # Display Result
    st.divider()
    st.subheader("Model Prediction")
    
    if prediction == 1:
        st.error(f"⚠️ FRAUD DETECTED (Confidence: {probability:.2%})")
    else:
        st.balloons()
        st.success(f"✅ Transaction Legitimate (Confidence: {1-probability:.2%})")

    # Feature Importance (Optional)
    with st.expander("See Technical Details (V1-V28 Features)"):
        st.write(transaction)

# Visualization Section
st.divider()
st.header("Model Performance Metrics")

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("Confusion Matrix")
    # You can load the image saved in Part 1, or replot it here if you have the data
    try:
        st.image("confusion_matrix.png", caption="Model Evaluation on Test Data")
    except:
        st.write("Run training script to generate matrix image.")

with col_viz2:
    st.subheader("Algorithm Details")
    st.markdown("""
    * **Algorithm:** XGBoost Classifier
    * **Preprocessing:** StandardScaler & SMOTE (for imbalance handling)
    * **Metrics Used:** Precision, Recall, F1-Score
    """)