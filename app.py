import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="BNPL Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #F4F7FB;
    color: #061A33;
}

.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1600px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001B3F 0%, #002B5C 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

div[role="radiogroup"] label {
    background: rgba(255,255,255,0.08);
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 8px;
    font-weight: 600;
}

div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.18);
}

/* Text */
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #061A33;
}

.subtitle {
    color: #5D6B82;
    font-size: 15px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 20px;
    font-weight: 800;
    color: #071D49;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* KPI */
.kpi-card {
    background: white;
    border: 1px solid #E0E8F3;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    min-height: 105px;
}

.kpi-label {
    font-size: 13px;
    color: #6B7280;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #061A33;
}

/* Cards */
.result-card {
    background: white;
    border: 1px solid #E0E8F3;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

.risk-low {
    background: #DDF8E8;
    color: #118D57;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}

.risk-medium {
    background: #FFF4CC;
    color: #B88400;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}

.risk-high {
    background: #FFE1E1;
    color: #D62828;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
}

.recommend-box {
    background: #EAF8F0;
    color: #0F5132;
    padding: 14px;
    border-radius: 12px;
    font-size: 14px;
}

/* Inputs */
label {
    color: #071D49 !important;
    font-weight: 600 !important;
}

input {
    background-color: white !important;
    color: #061A33 !important;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #061A33 !important;
}

.stButton > button {
    width: 100%;
    height: 48px;
    background: linear-gradient(90deg, #005BFF, #287DFF);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("best_bnpl_model.pkl")


@st.cache_data
def load_features():
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv("BNPL_dataset.csv")


def clean_feature_name(name):
    return (
        name.replace("_", " ")
        .replace("bnpl", "BNPL")
        .replace("dti", "DTI")
        .title()
    )


def get_model_importance(model, feature_columns):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "named_steps"):
        final_model = model.named_steps.get("model")
        if hasattr(final_model, "feature_importances_"):
            values = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            values = np.abs(final_model.coef_[0])
        else:
            values = np.ones(len(feature_columns))
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.ones(len(feature_columns))

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Feature Name": [clean_feature_name(x) for x in feature_columns],
        "Importance": values
    })

    return importance_df.sort_values("Importance", ascending=False)


def style_chart(fig, height=420, left_margin=120, bottom_margin=90):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#061A33", size=13),
        margin=dict(l=left_margin, r=40, t=60, b=bottom_margin),
        xaxis=dict(
            tickfont=dict(color="#061A33", size=12),
            titlefont=dict(color="#061A33", size=13),
            automargin=True
        ),
        yaxis=dict(
            tickfont=dict(color="#061A33", size=12),
            titlefont=dict(color="#061A33", size=13),
            automargin=True
        ),
        legend=dict(
            font=dict(color="#061A33")
        )
    )
    return fig


try:
    model = load_model()
    feature_columns = load_features()
except Exception as e:
    st.error("Failed to load model or feature columns.")
    st.exception(e)
    st.stop()

try:
    df = load_data()
except Exception:
    df = None


with st.sidebar:
    st.markdown("## 💳 BNPL RISK")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Predict Credit Risk",
            "Portfolio Overview",
            "Data Insights",
            "Model Explainability",
            "SHAP Analysis",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Model Information")
    st.write("Model Type")
    st.write(type(model).__name__)
    st.write("Dataset")
    st.write("BNPL Dataset")


st.markdown(
    '<div class="main-title">BNPL Credit Risk Prediction Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Predict customer default risk using machine learning</div>',
    unsafe_allow_html=True
)


if df is not None:
    total_customers = len(df)
    default_rate = df["default_flag"].mean() * 100
    avg_credit = df["credit_score"].mean()
    avg_income = df["monthly_income"].mean()
    high_risk = int((df["default_flag"] == 1).sum())

    k1, k2, k3, k4, k5 = st.columns(5)

    kpis = [
        ("👥 Total Customers", f"{total_customers:,}"),
        ("⚠️ Default Rate", f"{default_rate:.2f}%"),
        ("💳 Avg Credit Score", f"{avg_credit:.0f}"),
        ("💰 Avg Income", f"${avg_income:,.0f}"),
        ("🚨 High Risk Customers", f"{high_risk:,}")
    ]

    for col, (label, value) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("<br>", unsafe_allow_html=True)


# ==============================
# INPUT SECTION
# ==============================
st.markdown('<div class="section-title">Predict Credit Risk</div>', unsafe_allow_html=True)

left, middle, right = st.columns([2.1, 1.1, 1.1])

with left:
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 80, 30)
        employment_type = st.selectbox(
            "Employment Type",
            ["Employed", "Self-Employed", "Student", "Unemployed"]
        )
        purchase_amount = st.number_input("Purchase Amount", value=500.0)
        missed_payments = st.number_input("Missed Payments", value=0)
        location = st.selectbox(
            "Location",
            ["Australia", "Canada", "Germany", "India", "UK", "USA"]
        )

    with c2:
        monthly_income = st.number_input("Monthly Income", value=5000.0)
        debt_to_income_ratio = st.number_input("Debt to Income Ratio", value=0.30)
        bnpl_installments = st.selectbox(
            "BNPL Installments",
            [1, 2, 3, 4, 6, 8, 10, 12],
            index=3
        )
        risk_score = st.slider("Risk Score", 0.0, 1.0, 0.50)
        customer_segment = st.selectbox(
            "Customer Segment",
            ["High Risk", "Low Risk", "Medium Risk"]
        )

    with c3:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        app_usage_frequency = st.number_input("App Usage Frequency", value=10.0)
        repayment_delay_days = st.number_input("Repayment Delay Days", value=0)
        product_category = st.selectbox(
            "Product Category",
            ["Beauty", "Electronics", "Fashion", "Home", "Sports"]
        )
        transaction_date = st.date_input("Transaction Date")

    predict_button = st.button("Predict Risk")


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

try:
    probability = model.predict_proba(input_data)[0][1]
except Exception:
    probability = float(model.predict(input_data)[0])

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


with middle:
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-card">
            <p><b>Default Probability</b></p>
            <h1 style="text-align:center; color:#005BFF; font-size:48px;">{probability * 100:.2f}%</h1>
            <p><b>Risk Level</b></p>
            <div class="{risk_class}">{risk_label}</div>
            <br>
            <p><b>Recommendation</b></p>
            <div class="recommend-box">{recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown('<div class="section-title">Default Probability</div>', unsafe_allow_html=True)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 30, "color": "#061A33"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#005BFF"},
            "steps": [
                {"range": [0, 30], "color": "#DDF8E8"},
                {"range": [30, 60], "color": "#FFF4CC"},
                {"range": [60, 100], "color": "#FFE1E1"}
            ],
        }
    ))

    gauge.update_layout(
        height=300,
        paper_bgcolor="white",
        font=dict(color="#061A33"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(gauge, use_container_width=True)


# ==============================
# MODEL EXPLANATION
# ==============================
if page in ["Dashboard", "Model Explainability", "SHAP Analysis"]:
    st.markdown("---")

    importance_df = get_model_importance(model, feature_columns).head(10)

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("### Feature Importance")

        fig_imp = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature Name",
            orientation="h",
            color_discrete_sequence=["#005BFF"]
        )

        fig_imp = style_chart(fig_imp, height=460, left_margin=160, bottom_margin=100)
        fig_imp.update_xaxes(title_text="Importance")
        fig_imp.update_yaxes(title_text="")

        st.plotly_chart(fig_imp, use_container_width=True)

    with g2:
        st.markdown("### Local Explanation")

        local_df = importance_df.head(5).copy()
        local_df["Estimated Contribution"] = local_df["Importance"] * probability

        fig_local = px.bar(
            local_df.sort_values("Estimated Contribution"),
            x="Estimated Contribution",
            y="Feature Name",
            orientation="h",
            color_discrete_sequence=["#FF5A5F"]
        )

        fig_local = style_chart(fig_local, height=460, left_margin=160, bottom_margin=100)
        fig_local.update_xaxes(title_text="Estimated Contribution")
        fig_local.update_yaxes(title_text="")

        st.plotly_chart(fig_local, use_container_width=True)


# ==============================
# DATA INSIGHTS
# ==============================
if df is not None and page in ["Dashboard", "Portfolio Overview", "Data Insights"]:
    st.markdown("---")
    st.markdown("### Data Insights")

    d1, d2 = st.columns(2)

    with d1:
        fig_credit = px.histogram(
            df,
            x="credit_score",
            color="default_flag",
            title="Credit Score Distribution by Default Status"
        )

        fig_credit = style_chart(fig_credit, height=430, left_margin=90, bottom_margin=110)
        fig_credit.update_xaxes(title_text="Credit Score")
        fig_credit.update_yaxes(title_text="Count")

        st.plotly_chart(fig_credit, use_container_width=True)

    with d2:
        fig_delay = px.box(
            df,
            x="default_flag",
            y="repayment_delay_days",
            title="Repayment Delay by Default Status"
        )

        fig_delay = style_chart(fig_delay, height=430, left_margin=100, bottom_margin=110)
        fig_delay.update_xaxes(title_text="Default Flag")
        fig_delay.update_yaxes(title_text="Repayment Delay Days")

        st.plotly_chart(fig_delay, use_container_width=True)


st.markdown(
    """
    <div style="
        background:#FFF8DC;
        border:1px solid #F2D98D;
        border-radius:12px;
        padding:16px;
        margin-top:20px;
        color:#5C4A00;
        font-size:14px;
    ">
    This dashboard uses machine learning to predict BNPL default risk.
    Model explanations are used to support decision-making, not as the only decision factor.
    </div>
    """,
    unsafe_allow_html=True
)