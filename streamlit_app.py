import streamlit as st
import pandas as pd
import joblib


# Page Configuration
st.set_page_config(
    page_title="Credit Risk Assessment Tool",
    layout="wide"
)


# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("final_gradient_boosting_model.pkl")

try:
    model = load_model()
except Exception:
    st.error(
        "Model could not be loaded. Please ensure "
        "`final_gradient_boosting_model.pkl` is available."
    )
    st.stop()


# App Header
st.title("Credit Card Default Risk Assessment")

st.markdown(
    """
    This tool estimates the **probability that a customer will default on their next
    credit card payment** using a machine learning model trained on historical data.

    **Use cases**
    - Early identification of high-risk customers  
    - Credit limit review  
    - Manual risk assessment support
    """
)

st.divider()


# Input Form
st.subheader("Customer Information")

with st.form("risk_form"):

    st.markdown("### Core Details")
    col1, col2, col3 = st.columns(3)

    with col2:
        LIMIT_BAL = st.number_input(
            "Credit Limit (LIMIT_BAL)",
            min_value=1000,
            step=1000
        )

    col1b, col2b, col3b = st.columns(3)
    with col2b:
        AGE = st.number_input(
            "Age",
            min_value=18,
            max_value=100
        )

    # Repayment History (3 columns)
    st.markdown("### Repayment History (Previous Months)")
    r1, r2, r3 = st.columns(3)

    PAY_0 = r1.number_input("PAY_0", min_value=-2, max_value=9, value=0)
    PAY_2 = r2.number_input("PAY_2", min_value=-2, max_value=9, value=0)
    PAY_3 = r3.number_input("PAY_3", min_value=-2, max_value=9, value=0)

    r4, r5, r6 = st.columns(3)
    PAY_4 = r4.number_input("PAY_4", min_value=-2, max_value=9, value=0)
    PAY_5 = r5.number_input("PAY_5", min_value=-2, max_value=9, value=0)
    PAY_6 = r6.number_input("PAY_6", min_value=-2, max_value=9, value=0)


    # Bill Amounts (3 columns)
    st.markdown("### Bill Amounts (Last 6 Months)")
    b1, b2, b3 = st.columns(3)
    BILL_AMT1 = b1.number_input("BILL_AMT1", min_value=0, step=100)
    BILL_AMT2 = b2.number_input("BILL_AMT2", min_value=0, step=100)
    BILL_AMT3 = b3.number_input("BILL_AMT3", min_value=0, step=100)

    b4, b5, b6 = st.columns(3)
    BILL_AMT4 = b4.number_input("BILL_AMT4", min_value=0, step=100)
    BILL_AMT5 = b5.number_input("BILL_AMT5", min_value=0, step=100)
    BILL_AMT6 = b6.number_input("BILL_AMT6", min_value=0, step=100)

    BILL_AMTS = [
        BILL_AMT1, BILL_AMT2, BILL_AMT3,
        BILL_AMT4, BILL_AMT5, BILL_AMT6
    ]

    # Payment Amounts (3 columns)
    st.markdown("### Payment Amounts (Last 6 Months)")
    p1, p2, p3 = st.columns(3)
    PAY_AMT1 = p1.number_input("PAY_AMT1", min_value=0, step=100)
    PAY_AMT2 = p2.number_input("PAY_AMT2", min_value=0, step=100)
    PAY_AMT3 = p3.number_input("PAY_AMT3", min_value=0, step=100)

    p4, p5, p6 = st.columns(3)
    PAY_AMT4 = p4.number_input("PAY_AMT4", min_value=0, step=100)
    PAY_AMT5 = p5.number_input("PAY_AMT5", min_value=0, step=100)
    PAY_AMT6 = p6.number_input("PAY_AMT6", min_value=0, step=100)

    PAY_AMTS = [
        PAY_AMT1, PAY_AMT2, PAY_AMT3,
        PAY_AMT4, PAY_AMT5, PAY_AMT6
    ]

    st.markdown("")
    _, center_col, _ = st.columns([1, 1, 1])
    with center_col:
        submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True)



# Inference Logic
if submitted:

    if LIMIT_BAL <= 0:
        st.error("Credit limit must be greater than 0.")
        st.stop()

    if all(b == 0 for b in BILL_AMTS):
        st.warning(
            "All bill amounts are zero. Please verify the billing data."
        )


    # Prepare Input Data
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

    expected_features = model.feature_names_in_

    input_data = input_data.reindex(
        columns=expected_features,
        fill_value=0
    )


    # Prediction
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
    except Exception:
        st.error(
            "Prediction failed due to invalid input format."
        )
        st.stop()


    # Output Results
    st.divider()
    st.subheader("Risk Assessment Result")

    if prediction == 1:
        st.error("High Risk of Default")
    else:
        st.success("Low Risk of Default")

    st.metric(
        label="Probability of Default",
        value=f"{probability:.2%}"
    )

    st.markdown(
        """
        **Interpretation**
        - Higher probability indicates higher likelihood of default.
        - Intended for **risk ranking**, not automatic rejection.
        """
    )

    with st.expander("How this score should be used"):
        st.markdown(
            """
            - High-risk customers may be prioritised for manual review  
            - Credit limits can be adjusted proactively  
            - Supports risk-based decision-making
            """
        )
