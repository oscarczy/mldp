import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment Tool",
    page_icon="💳",
    layout="centered"
)

# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_gradient_boosting_model.pkl")

try:
    model = load_model()
except Exception:
    st.error(
        "Model could not be loaded. Please ensure the trained model file "
        "`final_gradient_boosting_model.pkl` is available."
    )
    st.stop()

# -------------------------------------------------
# App Header
# -------------------------------------------------
st.title("Credit Card Default Risk Assessment")
st.markdown(
    """
    This tool estimates the **probability that a customer will default on their next credit card payment**  
    using a machine learning model trained on historical repayment behaviour.

    **Use case:**  
    - Early identification of high-risk customers  
    - Credit limit review  
    - Manual risk assessment
    """
)

st.divider()

# -------------------------------------------------
# Input Form
# -------------------------------------------------
st.subheader("Customer Information")

with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        LIMIT_BAL = st.number_input("Credit Limit (LIMIT_BAL)", min_value=1000, step=1000)
        AGE = st.number_input("Age", min_value=18, max_value=100)

    with col2:
        PAY_0 = st.slider("Repayment Status Last Month (PAY_0)", -2, 9, 0)

    st.markdown("Repayment History (Previous Months)")
    c1, c2, c3, c4, c5 = st.columns(5)
    PAY_2 = c1.slider("PAY_2", -2, 9, 0)
    PAY_3 = c2.slider("PAY_3", -2, 9, 0)
    PAY_4 = c3.slider("PAY_4", -2, 9, 0)
    PAY_5 = c4.slider("PAY_5", -2, 9, 0)
    PAY_6 = c5.slider("PAY_6", -2, 9, 0)

    st.markdown("Bill Amounts (Last 6 Months)")
    BILL_AMTS = [
        st.number_input(f"BILL_AMT{i}", min_value=0, step=100)
        for i in range(1, 7)
    ]

    st.markdown("Payment Amounts (Last 6 Months)")
    PAY_AMTS = [
        st.number_input(f"PAY_AMT{i}", min_value=0, step=100)
        for i in range(1, 7)
    ]

    submitted = st.form_submit_button("Assess Credit Risk")

# -------------------------------------------------
# Inference Logic
# -------------------------------------------------
if submitted:

    # Basic validation
    if LIMIT_BAL <= 0:
        st.error("Credit limit must be greater than 0.")
        st.stop()

    if all(b == 0 for b in BILL_AMTS):
        st.warning("All bill amounts are zero. Please verify the customer billing data.")

    # -------------------------------------------------
    # Prepare Input Data
    # -------------------------------------------------
    input_data = pd.DataFrame({
        "LIMIT_BAL": [LIMIT_BAL],
        "AGE": [AGE],
        "PAY_0": [PAY_0],
        "PAY_2": [PAY_2],
        "PAY_3": [PAY_3],
        "PAY_4": [PAY_4],
        "PAY_5": [PAY_5],
        "PAY_6": [PAY_6],
        **{f"BILL_AMT{i+1}": [BILL_AMTS[i]] for i in range(6)},
        **{f"PAY_AMT{i+1}": [PAY_AMTS[i]] for i in range(6)},
    })

    # -------------------------------------------------
    # 🔒 Align input schema with model expectations
    # -------------------------------------------------
    expected_features = model.feature_names_in_

    input_data = input_data.reindex(
        columns=expected_features,
        fill_value=0
    )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
    except Exception:
        st.error(
            "❌ Prediction failed due to invalid input format. "
            "Please ensure all fields are filled correctly."
        )
        st.stop()

    # -------------------------------------------------
    # Output Results
    # -------------------------------------------------
    st.divider()
    st.subheader("Risk Assessment Result")

    if prediction == 1:
        st.error("**High Risk of Default**")
    else:
        st.success("**Low Risk of Default**")

    st.metric(
        label="Probability of Default",
        value=f"{probability:.2%}"
    )

    st.markdown(
        """
        **Interpretation:**
        - Higher probability indicates higher likelihood of default.
        - This score can be used for **risk ranking**, not just binary decisions.
        """
    )

    with st.expander("How this score should be used"):
        st.markdown(
            """
            - Customers with **higher default probability** may be prioritised for manual review.
            - Credit limits may be adjusted proactively to reduce potential losses.
            - The model supports **risk-based decision-making**, not automatic rejection.
            """
        )