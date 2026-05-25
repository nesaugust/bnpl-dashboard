import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="BNPL Credit Risk Prediction Dashboard",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #061A33 0%, #082B57 50%, #03101F 100%);
    color: #EAF6FF;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #082B57 0%, #031A35 100%);
    border-right: 1px solid rgba(255,255,255,0.12);
}
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: white;
}
.subtitle {
    font-size: 16px;
    color: #BFDFFF;
    margin-bottom: 25px;
}
.card {
    background: rgba(13, 52, 96, 0.75);
    border: 1px solid rgba(95, 180, 255, 0.25);
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.risk-low {
    color: #2ECC71;
    font-size: 32px;
    font-weight: 800;
}
.risk-medium {
    color: #F1C40F;
    font-size: 32px;
    font-weight: 800;
}
.risk-high {
    color: #E74C3C;
    font-size: 32px;
    font-weight: 800;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0B3D91, #0077FF);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,119,255,0.28);
}
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #D9ECFF !important;
}
.stButton > button {
    background: linear-gradient(90deg, #00A6FB, #3949FF);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 700;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("best_bnpl_model.pkl")


@st.cache_data
def load_feature_columns():
    with open("feature_columns.json", "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_data():
    return pd.read_csv("BNPL_dataset.csv")


try:
    model = load_model()
    feature_columns = load_feature_columns()
except Exception as e:
    st.error("App failed to load the model or feature columns.")
    st.exception(e)
    st.stop()


try:
    df = load_data()
except Exception as e:
    df = None
    st.warning("BNPL_dataset.csv was not found or could not be loaded. Dashboard insights will be hidden.")


st.markdown(
    '<div class="main-title">BNPL Credit Risk Prediction Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Predict customer default risk using machine learning and financial behaviour indicators.</div>',
    unsafe_allow_html=True
)


if df is not None:
    try:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Average Credit Score", f"{df['credit_score'].mean():.0f}")
        col3.metric("Average Income", f"${df['monthly_income'].mean():,.0f}")
        col4.metric("Default Rate", f"{df['default_flag'].mean() * 100:.2f}%")

        st.divider()
    except Exception as e:
        st.warning("Dataset loaded, but KPI section could not be displayed.")
        st.exception(e)


st.sidebar.markdown("## 💳 Credit Risk Dashboard")
st.sidebar.markdown("### Customer Input")

employment_type = st.sidebar.selectbox(
    "Employment Type",
    ["Employed", "Self-Employed", "Student", "Unemployed"]
)

product_category = st.sidebar.selectbox(
    "Product Category",
    ["Beauty", "Electronics", "Fashion", "Home", "Sports"]
)

location = st.sidebar.selectbox(
    "Location",
    ["Australia", "Canada", "Germany", "India", "UK", "USA"]
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment",
    ["High Risk", "Low Risk", "Medium Risk"]
)

transaction_date = st.sidebar.date_input("Transaction Date")

transaction_month = transaction_date.month
transaction_year = transaction_date.year
transaction_day = transaction_date.day


st.subheader("Customer Financial Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

with col2:
    purchase_amount = st.number_input("Purchase Amount", min_value=0.0, value=500.0)
    bnpl_installments = st.slider("BNPL Installments", 1, 12, 4)
    app_usage_frequency = st.number_input("App Usage Frequency", min_value=0.0, value=10.0)

with col3:
    repayment_delay_days = st.number_input("Repayment Delay Days", min_value=0, value=0)
    missed_payments = st.number_input("Missed Payments", min_value=0, value=0)
    debt_to_income_ratio = st.number_input(
        "Debt to Income Ratio",
        min_value=0.0,
        max_value=5.0,
        value=0.30
    )

risk_score = st.slider("Risk Score", 0.0, 1.0, 0.50)


input_data = pd.DataFrame({
    "age": [age],
    "monthly_income": [monthly_income],
    "credit_score": [credit_score],
    "purchase_amount": [purchase_amount],
    "bnpl_installments": [bnpl_installments],
    "repayment_delay_days": [repayment_delay_days],
    "missed_payments": [missed_payments],
    "app_usage_frequency": [app_usage_frequency],
    "debt_to_income_ratio": [debt_to_income_ratio],
    "risk_score": [risk_score],
    "transaction_month": [transaction_month],
    "transaction_year": [transaction_year],
    "transaction_day": [transaction_day],
    "employment_type_Self-Employed": [1 if employment_type == "Self-Employed" else 0],
    "employment_type_Student": [1 if employment_type == "Student" else 0],
    "employment_type_Unemployed": [1 if employment_type == "Unemployed" else 0],
    "product_category_Electronics": [1 if product_category == "Electronics" else 0],
    "product_category_Fashion": [1 if product_category == "Fashion" else 0],
    "product_category_Home": [1 if product_category == "Home" else 0],
    "product_category_Sports": [1 if product_category == "Sports" else 0],
    "location_Canada": [1 if location == "Canada" else 0],
    "location_Germany": [1 if location == "Germany" else 0],
    "location_India": [1 if location == "India" else 0],
    "location_UK": [1 if location == "UK" else 0],
    "location_USA": [1 if location == "USA" else 0],
    "customer_segment_Low Risk": [1 if customer_segment == "Low Risk" else 0],
    "customer_segment_Medium Risk": [1 if customer_segment == "Medium Risk" else 0],
})

for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_columns]


st.divider()
st.subheader("Prediction Result")

if st.button("Predict Credit Risk"):
    try:
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = prediction

        if probability < 0.30:
            risk_label = "Low Risk"
            risk_class = "risk-low"
            recommendation = "Customer is likely safe for BNPL approval."
        elif probability < 0.60:
            risk_label = "Medium Risk"
            risk_class = "risk-medium"
            recommendation = "Customer requires additional review before approval."
        else:
            risk_label = "High Risk"
            risk_class = "risk-high"
            recommendation = "Customer has high default risk. Consider rejecting or lowering credit limit."

        st.markdown(
            f"""
            <div class="card">
                <p>Predicted Default Probability</p>
                <h1>{probability * 100:.2f}%</h1>
                <p class="{risk_class}">{risk_label}</p>
                <p>{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            prob_df = pd.DataFrame({
                "Status": ["No Default", "Default"],
                "Probability": [1 - probability, probability]
            })

            fig_prob = px.bar(
                prob_df,
                x="Status",
                y="Probability",
                title="Default Probability",
                text="Probability",
                template="plotly_dark"
            )

            fig_prob.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig_prob.update_layout(
                paper_bgcolor="#061A33",
                plot_bgcolor="#061A33",
                font=dict(color="white"),
                yaxis_tickformat=".0%"
            )

            st.plotly_chart(fig_prob, use_container_width=True)

        with col2:
            risk_factors = pd.DataFrame({
                "Factor": [
                    "Credit Score",
                    "Debt to Income Ratio",
                    "Missed Payments",
                    "Repayment Delay Days",
                    "Purchase Amount"
                ],
                "Value": [
                    credit_score,
                    debt_to_income_ratio,
                    missed_payments,
                    repayment_delay_days,
                    purchase_amount
                ]
            })

            fig_factors = px.bar(
                risk_factors,
                x="Value",
                y="Factor",
                orientation="h",
                title="Customer Risk Indicators",
                template="plotly_dark"
            )

            fig_factors.update_layout(
                paper_bgcolor="#061A33",
                plot_bgcolor="#061A33",
                font=dict(color="white")
            )

            st.plotly_chart(fig_factors, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)


if df is not None:
    st.divider()
    st.subheader("Dashboard Insights")

    try:
        col1, col2 = st.columns(2)

        with col1:
            fig_credit = px.histogram(
                df,
                x="credit_score",
                color="default_flag",
                title="Credit Score Distribution by Default Status",
                template="plotly_dark"
            )

            fig_credit.update_layout(
                paper_bgcolor="#061A33",
                plot_bgcolor="#061A33",
                font=dict(color="white")
            )

            st.plotly_chart(fig_credit, use_container_width=True)

        with col2:
            fig_delay = px.box(
                df,
                x="default_flag",
                y="repayment_delay_days",
                title="Repayment Delay by Default Status",
                template="plotly_dark"
            )

            fig_delay.update_layout(
                paper_bgcolor="#061A33",
                plot_bgcolor="#061A33",
                font=dict(color="white")
            )

            st.plotly_chart(fig_delay, use_container_width=True)

    except Exception as e:
        st.warning("Dashboard insights could not be displayed.")
        st.exception(e)


st.markdown("---")
st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        color:#BFDFFF;
        font-size:14px;
    ">
        BNPL Credit Risk Prediction Dashboard © 2026<br>
        <span style="color:white; font-size:18px; font-weight:700;">
            Created by Agnes Jeni Makay
        </span>
    </div>
    """,
    unsafe_allow_html=True
)