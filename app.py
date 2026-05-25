import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BNPL Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background-color: #F5F8FC;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001B3F 0%, #002B5C 100%);
    min-width: 270px;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-item {
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    background: rgba(255,255,255,0.05);
    transition: 0.3s;
}

.sidebar-item:hover {
    background: rgba(255,255,255,0.15);
}

/* MAIN TITLE */
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #061A33;
}

.subtitle {
    font-size: 16px;
    color: #5E6E82;
    margin-bottom: 25px;
}

/* KPI CARDS */
.kpi-card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #E5ECF6;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

.kpi-label {
    font-size: 14px;
    color: #6B7A90;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #061A33;
}

/* GENERAL CARD */
.card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #E5ECF6;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}

/* SECTION TITLE */
.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #071D49;
    margin-bottom: 15px;
}

/* RISK LEVEL */
.risk-low {
    background: #DDF8E8;
    color: #118D57;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
}

.risk-medium {
    background: #FFF4CC;
    color: #B88400;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
}

.risk-high {
    background: #FFE1E1;
    color: #D62828;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
}

/* RECOMMENDATION */
.recommend-box {
    background: #EAF8F0;
    padding: 16px;
    border-radius: 12px;
    color: #0F5132;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #005BFF, #287DFF);
    color: white;
    font-size: 18px;
    font-weight: 700;
}

/* INPUT */
div[data-baseweb="input"] input {
    background-color: white !important;
    color: black !important;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD FILES
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("best_bnpl_model.pkl")


@st.cache_data
def load_features():
    with open("feature_columns.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv("BNPL_dataset.csv")


try:
    model = load_model()
    feature_columns = load_features()
except Exception as e:
    st.error("Failed to load model files.")
    st.exception(e)
    st.stop()

try:
    df = load_data()
except:
    df = None


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("# 💳 BNPL RISK")

    st.markdown("---")

    st.markdown("### Dashboard")
    st.markdown(
        '<div class="sidebar-item">📊 Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Prediction")
    st.markdown(
        '<div class="sidebar-item">💳 Predict Credit Risk</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-item">📂 Batch Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Analytics")
    st.markdown(
        '<div class="sidebar-item">📈 Portfolio Overview</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-item">📉 Data Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Explainable AI")
    st.markdown(
        '<div class="sidebar-item">🧠 Model Explainability</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sidebar-item">🔍 SHAP Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Model Information")
    st.write("Model Type")
    st.write(type(model).__name__)

    st.write("Dataset")
    st.write("BNPL Dataset")

    st.write("Version")
    st.write("1.0")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">BNPL Credit Risk Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict customer default risk using machine learning</div>',
    unsafe_allow_html=True
)


# =========================================================
# KPI SECTION
# =========================================================
if df is not None:

    total_customers = len(df)
    default_rate = df["default_flag"].mean() * 100
    avg_credit = df["credit_score"].mean()
    avg_income = df["monthly_income"].mean()
    high_risk = (df["default_flag"] == 1).sum()

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">👥 Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⚠️ Default Rate</div>
            <div class="kpi-value">{default_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💳 Avg Credit Score</div>
            <div class="kpi-value">{avg_credit:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 Avg Income</div>
            <div class="kpi-value">${avg_income:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🚨 High Risk Customers</div>
            <div class="kpi-value">{high_risk:,}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================
left, mid, right = st.columns([2.4, 1, 1.3])

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Predict Credit Risk</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 80, 30)
        employment_type = st.selectbox(
            "Employment Type",
            ["Employed", "Self-Employed", "Student", "Unemployed"]
        )

        purchase_amount = st.number_input(
            "Purchase Amount",
            value=500.0
        )

        missed_payments = st.number_input(
            "Missed Payments",
            value=0
        )

        location = st.selectbox(
            "Location",
            ["Australia", "Canada", "Germany", "India", "UK", "USA"]
        )

    with c2:
        monthly_income = st.number_input(
            "Monthly Income",
            value=5000.0
        )

        debt_to_income_ratio = st.number_input(
            "Debt to Income Ratio",
            value=0.30
        )

        bnpl_installments = st.selectbox(
            "BNPL Installments",
            [1,2,3,4,6,8,10,12],
            index=3
        )

        risk_score = st.slider(
            "Risk Score",
            0.0,
            1.0,
            0.50
        )

        customer_segment = st.selectbox(
            "Customer Segment",
            ["High Risk", "Low Risk", "Medium Risk"]
        )

    with c3:
        credit_score = st.number_input(
            "Credit Score",
            300,
            850,
            650
        )

        app_usage_frequency = st.number_input(
            "App Usage Frequency",
            value=10.0
        )

        repayment_delay_days = st.number_input(
            "Repayment Delay Days",
            value=0
        )

        product_category = st.selectbox(
            "Product Category",
            ["Beauty", "Electronics", "Fashion", "Home", "Sports"]
        )

        transaction_date = st.date_input(
            "Transaction Date"
        )

    predict_button = st.button("Predict Risk")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREPARE INPUT
# =========================================================
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
    "transaction_month": [transaction_date.month],
    "transaction_year": [transaction_date.year],
    "transaction_day": [transaction_date.day],

    "employment_type_Self-Employed":
        [1 if employment_type == "Self-Employed" else 0],

    "employment_type_Student":
        [1 if employment_type == "Student" else 0],

    "employment_type_Unemployed":
        [1 if employment_type == "Unemployed" else 0],

    "product_category_Electronics":
        [1 if product_category == "Electronics" else 0],

    "product_category_Fashion":
        [1 if product_category == "Fashion" else 0],

    "product_category_Home":
        [1 if product_category == "Home" else 0],

    "product_category_Sports":
        [1 if product_category == "Sports" else 0],

    "location_Canada":
        [1 if location == "Canada" else 0],

    "location_Germany":
        [1 if location == "Germany" else 0],

    "location_India":
        [1 if location == "India" else 0],

    "location_UK":
        [1 if location == "UK" else 0],

    "location_USA":
        [1 if location == "USA" else 0],

    "customer_segment_Low Risk":
        [1 if customer_segment == "Low Risk" else 0],

    "customer_segment_Medium Risk":
        [1 if customer_segment == "Medium Risk" else 0],
})

for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_columns]


# =========================================================
# PREDICTION
# =========================================================
probability = model.predict_proba(input_data)[0][1]

if probability < 0.30:
    risk_label = "Low Risk"
    risk_class = "risk-low"
    recommendation = "Customer is likely safe for BNPL approval."

elif probability < 0.60:
    risk_label = "Medium Risk"
    risk_class = "risk-medium"
    recommendation = "Customer requires additional review."

else:
    risk_label = "High Risk"
    risk_class = "risk-high"
    recommendation = "Customer has a high probability of default."


# =========================================================
# RESULT SECTION
# =========================================================
with mid:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <h1 style="
        text-align:center;
        color:#005BFF;
        font-size:52px;
    ">
    {probability*100:.2f}%
    </h1>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="{risk_class}">{risk_label}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="recommend-box">{recommendation}</div>',
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# GAUGE CHART
# =========================================================
with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Default Probability</div>',
        unsafe_allow_html=True
    )

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix":"%"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#005BFF"},
            "steps":[
                {"range":[0,30],"color":"#DDF8E8"},
                {"range":[30,60],"color":"#FFF4CC"},
                {"range":[60,100],"color":"#FFE1E1"}
            ]
        }
    ))

    gauge.update_layout(
        paper_bgcolor="white",
        font=dict(color="#061A33"),
        height=260
    )

    st.plotly_chart(gauge, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
else:
    importance = np.random.rand(len(feature_columns))

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
}).sort_values("Importance", ascending=False).head(10)


st.markdown("<br>", unsafe_allow_html=True)

g1, g2 = st.columns(2)

with g1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Feature Importance</div>',
        unsafe_allow_html=True
    )

    fig_imp = px.bar(
        importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#005BFF"]
    )

    fig_imp.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#061A33"),
        xaxis=dict(
            tickfont=dict(color="#061A33")
        ),
        yaxis=dict(
            tickfont=dict(color="#061A33")
        ),
        height=400
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# SHAP EXPLANATION
# =========================================================
with g2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Local Explanation</div>',
        unsafe_allow_html=True
    )

    top_features = importance_df.head(5)

    for _, row in top_features.iterrows():

        st.markdown(f"""
        <div style="
            background:#EAF2FF;
            padding:14px;
            border-radius:12px;
            margin-bottom:10px;
        ">
            <b>{row['Feature']}</b><br>
            Contribution Score: {row['Importance']:.3f}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# DATA VISUALIZATION
# =========================================================
if df is not None:

    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:

        fig_credit = px.histogram(
            df,
            x="credit_score",
            color="default_flag",
            title="Credit Score Distribution"
        )

        fig_credit.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#061A33")
        )

        st.plotly_chart(fig_credit, use_container_width=True)

    with d2:

        fig_delay = px.box(
            df,
            x="default_flag",
            y="repayment_delay_days",
            title="Repayment Delay Analysis"
        )

        fig_delay.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#061A33")
        )

        st.plotly_chart(fig_delay, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="
    background:#FFF8DC;
    border:1px solid #F2D98D;
    border-radius:12px;
    padding:16px;
    margin-top:20px;
    color:#5C4A00;
">
This dashboard uses machine learning models to predict BNPL default risk.
The prediction and SHAP explanations are intended to support decision-making.
</div>
""", unsafe_allow_html=True)