import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="BNPL Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(37,99,235,0.10), transparent 28%),
        linear-gradient(135deg, #EEF4FC 0%, #F8FBFF 45%, #EAF1FA 100%);
    color: #071D49;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1500px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #111827 65%, #020617 100%);
    border-right: 1px solid rgba(148,163,184,0.25);
}

section[data-testid="stSidebar"] * {
    color: #E5E7EB !important;
}

.sidebar-list {
    background: rgba(30,41,59,0.75);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 16px;
    padding: 14px 16px;
    line-height: 2.1;
    font-weight: 700;
}

/* Hero */
.hero-section {
    text-align: center;
    padding: 36px 24px 32px 24px;
    margin-bottom: 28px;
    border-radius: 30px;
    background:
        radial-gradient(circle at 20% 20%, rgba(96,165,250,0.20), transparent 30%),
        radial-gradient(circle at 80% 10%, rgba(37,99,235,0.12), transparent 28%),
        linear-gradient(135deg, rgba(255,255,255,0.95), rgba(239,246,255,0.92));
    border: 1px solid rgba(191,219,254,0.85);
    box-shadow: 0 22px 60px rgba(15,23,42,0.08);
}

.main-title {
    font-size: 44px;
    font-weight: 900;
    color: #071D49;
    letter-spacing: -1.1px;
    margin-bottom: 10px;
}

.subtitle {
    color: #475569;
    font-size: 17px;
    font-weight: 500;
}

/* Section title */
.section-title {
    font-size: 24px;
    font-weight: 850;
    color: #071D49;
    margin-bottom: 20px;
    text-align: center;
}

/* Cards */
.kpi-card,
.result-card,
.info-card {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
    border: 1px solid #D6E4F5;
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 18px 45px rgba(15,23,42,0.08);
}

.info-card {
    min-height: 560px;
}

.result-card {
    min-height: 560px;
}

/* KPI */
.kpi-label {
    font-size: 12px;
    color: #64748B;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.kpi-value {
    font-size: 30px;
    font-weight: 900;
    color: #071D49;
    margin-top: 8px;
}

/* Risk labels */
.risk-low {
    background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
    color: #166534;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 28px;
    font-weight: 900;
    border: 1px solid #86EFAC;
}

.risk-medium {
    background: linear-gradient(135deg, #FEF3C7, #FDE68A);
    color: #92400E;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 28px;
    font-weight: 900;
    border: 1px solid #FCD34D;
}

.risk-high {
    background: linear-gradient(135deg, #FEE2E2, #FECACA);
    color: #991B1B;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 28px;
    font-weight: 900;
    border: 1px solid #FCA5A5;
}

.recommend-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: #1E40AF;
    padding: 16px;
    border-radius: 16px;
    font-size: 15px;
    line-height: 1.5;
}

/* Inputs */
label {
    color: #0F172A !important;
    font-weight: 750 !important;
    font-size: 14px !important;
}

input,
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #071D49 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 14px !important;
    min-height: 46px;
}

div[data-baseweb="select"] span {
    color: #071D49 !important;
}

details {
    background: #F8FAFC !important;
    border: 1px solid #D6E4F5 !important;
    border-radius: 16px !important;
    padding: 10px 14px !important;
}

/* Button */
.stButton > button {
    width: 220px;
    height: 52px;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white !important;
    border: none;
    border-radius: 14px;
    font-weight: 850;
    box-shadow: 0 12px 28px rgba(37,99,235,0.28);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1E40AF, #1D4ED8);
}

/* Charts */
.js-plotly-plot {
    border-radius: 22px;
    overflow: hidden;
    border: 1px solid #D6E4F5;
    box-shadow: 0 16px 36px rgba(15,23,42,0.08);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
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
    with open("feature_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data():
    return pd.read_csv("BNPL_dataset.csv")


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


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def clean_feature_name(name):
    return (
        name.replace("_", " ")
        .replace("bnpl", "BNPL")
        .replace("usa", "USA")
        .replace("uk", "UK")
        .title()
    )


def get_base_model(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def get_model_importance(model, feature_columns):
    base_model = get_base_model(model)

    if hasattr(base_model, "feature_importances_"):
        values = base_model.feature_importances_
    elif hasattr(base_model, "coef_"):
        values = np.abs(base_model.coef_[0])
    else:
        values = np.ones(len(feature_columns))

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Feature Name": [clean_feature_name(x) for x in feature_columns],
        "Importance": values
    })

    return importance_df.sort_values("Importance", ascending=False)


def style_chart(fig, height=420, left_margin=130, bottom_margin=100):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#061A33", size=13),
        margin=dict(l=left_margin, r=40, t=70, b=bottom_margin),
        legend=dict(font=dict(color="#061A33"))
    )

    fig.update_xaxes(
        tickfont=dict(color="#061A33", size=12),
        title_font=dict(color="#061A33", size=13),
        automargin=True
    )

    fig.update_yaxes(
        tickfont=dict(color="#061A33", size=12),
        title_font=dict(color="#061A33", size=13),
        automargin=True
    )

    return fig


def predict_probability(model, input_data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_data)[0][1]
    return float(model.predict(input_data)[0])


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 💳 BNPL RISK")
    st.markdown("---")

    page = "Dashboard"

    st.markdown("### Sections")
    st.markdown("""
    <div class="sidebar-list">
        <div>▸ Overview KPIs</div>
        <div>▸ Credit Risk Prediction</div>
        <div>▸ Data Insights</div>
        <div>▸ Explainable AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Information")
    st.write("Model Type")
    st.write(type(get_base_model(model)).__name__)
    st.write("Dataset")
    st.write("BNPL Dataset")
    st.write("Version")
    st.write("1.0")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-section">
        <div class="main-title">BNPL Credit Risk Prediction Dashboard</div>
        <div class="subtitle">
            Predict customer default risk using machine learning and explainable AI.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# KPI SECTION
# =========================================================
if df is not None and page in ["Dashboard", "Analytics"]:
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


# =========================================================
# INPUT SECTION
# =========================================================
if page in ["Dashboard", "Prediction", "Explainable AI"]:

    title_left, title_right = st.columns([1.7, 1])

    with title_left:
        st.markdown('<div class="section-title">Predict Credit Risk</div>', unsafe_allow_html=True)

    with title_right:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    left, middle = st.columns([1.7, 1])

    with left:
        with st.container(border=True):
            c1, c2 = st.columns(2)

            with c1:
                monthly_income = st.number_input("Monthly Income", value=5000.0)
                credit_score = st.number_input("Credit Score", 300, 850, 650)
                debt_to_income_ratio = st.number_input("Debt to Income Ratio", value=0.30)
                missed_payments = st.number_input("Missed Payments", value=0)

            with c2:
                purchase_amount = st.number_input("Purchase Amount", value=500.0)
                bnpl_installments = st.selectbox("BNPL Installments", [1, 2, 3, 4, 6, 8, 10, 12], index=3)
                repayment_delay_days = st.number_input("Repayment Delay Days", value=0)
                app_usage_frequency = st.number_input("App Usage Frequency", value=10.0)

            with st.expander("Advanced customer details"):
                c3, c4, c5 = st.columns(3)

                with c3:
                    age = st.number_input("Age", 18, 80, 30)
                    employment_type = st.selectbox(
                        "Employment Type",
                        ["Employed", "Self-Employed", "Student", "Unemployed"]
                    )

                with c4:
                    location = st.selectbox(
                        "Location",
                        ["Australia", "Canada", "Germany", "India", "UK", "USA"]
                    )
                    product_category = st.selectbox(
                        "Product Category",
                        ["Beauty", "Electronics", "Fashion", "Home", "Sports"]
                    )

                with c5:
                    customer_segment = st.selectbox(
                        "Customer Segment",
                        ["Low Risk", "Medium Risk", "High Risk"]
                    )
                    transaction_date = st.date_input("Transaction Date")

            risk_score = 0.50
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

    probability = predict_probability(model, input_data)

    # =========================================================
    # BUSINESS RULE ADJUSTMENT + EXPLANATION
    # =========================================================

    base_probability = probability
    explanations = []

    purchase_income_ratio = purchase_amount / max(monthly_income, 1)
    installment_amount = purchase_amount / max(bnpl_installments, 1)
    installment_income_ratio = installment_amount / max(monthly_income, 1)

    risk_adjustment = 0

    if monthly_income < 500:
        risk_adjustment += 0.25
        explanations.append("Very low monthly income increased default risk.")

    if purchase_income_ratio > 1:
        risk_adjustment += 0.25
        explanations.append(
            f"Purchase amount is {purchase_income_ratio:.1f}× higher than monthly income."
        )
    elif purchase_income_ratio > 0.5:
        risk_adjustment += 0.15
        explanations.append("Purchase amount is high compared to monthly income.")
    elif purchase_income_ratio > 0.3:
        risk_adjustment += 0.08
        explanations.append("Purchase amount is moderately high compared to monthly income.")

    if installment_income_ratio > 0.5:
        risk_adjustment += 0.20
        explanations.append("Installment amount consumes more than 50% of monthly income.")
    elif installment_income_ratio > 0.3:
        risk_adjustment += 0.10
        explanations.append("Installment amount consumes more than 30% of monthly income.")

    if credit_score < 500:
        risk_adjustment += 0.20
        explanations.append("Very low credit score increased default risk.")
    elif credit_score < 650:
        risk_adjustment += 0.10
        explanations.append("Below-average credit score slightly increased default risk.")

    if missed_payments > 0:
        missed_payment_adjustment = min(missed_payments * 0.10, 0.30)
        risk_adjustment += missed_payment_adjustment
        explanations.append(
            f"{missed_payments} missed payment(s) increased default risk."
        )

    if repayment_delay_days > 30:
        risk_adjustment += 0.25
        explanations.append("Repayment delay above 30 days significantly increased risk.")
    elif repayment_delay_days > 7:
        risk_adjustment += 0.15
        explanations.append("Repayment delay above 7 days increased risk.")
    elif repayment_delay_days > 0:
        risk_adjustment += 0.08
        explanations.append("Minor repayment delay slightly increased risk.")

    probability = min(base_probability + risk_adjustment, 0.99)

    if len(explanations) == 0:
        explanations.append("No additional business-rule adjustments were applied.")

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

    # Advanced customer details
    if age >= 60:
        risk_adjustment += 0.05
        explanations.append("Older customer age slightly increased default risk.")
    elif age < 21:
        risk_adjustment += 0.05
        explanations.append("Very young customer age slightly increased default risk.")
    
    if employment_type == "Self-Employed":
        risk_adjustment += 0.08
        explanations.append("Self-employed status slightly increased default risk.")
    elif employment_type == "Student":
        risk_adjustment += 0.05
        explanations.append("Student status slightly increased default risk.")
    elif employment_type == "Unemployed":
        risk_adjustment += 0.25
        explanations.append("Unemployed status significantly increased default risk.")

    if customer_segment == "High Risk":
        risk_adjustment += 0.20
        explanations.append("Customer segment is marked as High Risk.")
    elif customer_segment == "Medium Risk":
        risk_adjustment += 0.10
        explanations.append("Customer segment is marked as Medium Risk.")

    if product_category == "Electronics":
        risk_adjustment += 0.05
        explanations.append("Electronics purchases slightly increased default risk.")
    elif product_category == "Beauty":
        risk_adjustment += 0.03
        explanations.append("Beauty purchases slightly increased default risk.")

    if location in ["India"]:
        risk_adjustment += 0.03
        explanations.append("Location-based risk adjustment was applied.")

    # Transaction date logic
    if transaction_date.month in [11, 12]:
        risk_adjustment += 0.03
        explanations.append("End-of-year transaction period slightly increased risk.")

    explanation_html = "".join([f"<li>{exp}</li>" for exp in explanations])

    with middle:
        with st.container(border=True):

            st.markdown("**Default Probability**")

            st.markdown(
                f"<h1 style='text-align:center;color:#2563EB;font-size:52px'>{probability*100:.2f}%</h1>",
                unsafe_allow_html=True,
            )

            st.caption(
                f"Base Model: {base_probability*100:.2f}%   |   "
                f"Business Rules: +{risk_adjustment*100:.2f}%"
            )

            st.markdown("**Risk Level**")

            st.markdown(
                f'<div class="{risk_class}">{risk_label}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("**Recommendation**")

            st.markdown(
                f'<div class="recommend-box">{recommendation}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("**Adjustment Explanation**")

            st.markdown(
                "<div class='recommend-box'>"
                + "".join([f"• {e}<br>" for e in explanations])
                + "</div>",
                unsafe_allow_html=True,
            )
    
# =========================================================
# ANALYTICS PAGE
# =========================================================
if df is not None and page in ["Dashboard", "Analytics"]:
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


# =========================================================
# EXPLAINABLE AI PAGE
# =========================================================
if page in ["Dashboard", "Explainable AI"]:
    st.markdown("---")
    st.markdown("### Explainable AI")

    importance_df = get_model_importance(model, feature_columns).head(10)

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("#### Global Feature Importance")

        fig_imp = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature Name",
            orientation="h",
            color_discrete_sequence=["#005BFF"],
            title="Top Drivers of Model Prediction"
        )

        fig_imp = style_chart(fig_imp, height=470, left_margin=170, bottom_margin=110)
        fig_imp.update_xaxes(title_text="Importance")
        fig_imp.update_yaxes(title_text="")

        st.plotly_chart(fig_imp, use_container_width=True)

    with g2:
        st.markdown("#### SHAP-Style Impact Explanation")

        shap_df = importance_df.head(8).copy()
        shap_df["Impact"] = shap_df["Importance"] * probability

        lower_risk_features = [
            "credit_score",
            "monthly_income",
            "app_usage_frequency"
        ]

        shap_df["Direction"] = np.where(
            shap_df["Feature"].isin(lower_risk_features),
            "Lower Risk",
            "Higher Risk"
        )

        shap_df["Signed Impact"] = np.where(
            shap_df["Direction"] == "Lower Risk",
            -shap_df["Impact"],
            shap_df["Impact"]
        )

        fig_shap = px.bar(
            shap_df.sort_values("Signed Impact"),
            x="Signed Impact",
            y="Feature Name",
            orientation="h",
            color="Direction",
            color_discrete_map={
                "Higher Risk": "#FF5A5F",
                "Lower Risk": "#1D6BFF"
            },
            title="Estimated Impact on Default Probability"
        )

        fig_shap.add_vline(
            x=0,
            line_width=2,
            line_dash="dash",
            line_color="#061A33"
        )

        fig_shap = style_chart(fig_shap, height=470, left_margin=170, bottom_margin=110)
        fig_shap.update_xaxes(title_text="Impact on Default Probability")
        fig_shap.update_yaxes(title_text="")

        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown(
        """
        <div style="
            background:#EAF2FF;
            border:1px solid #BBD3FF;
            border-radius:12px;
            padding:16px;
            margin-top:15px;
            color:#123B73;
            font-size:14px;
        ">
        <b>Explanation:</b> Global Feature Importance shows which variables are most influential overall.
        The SHAP-style chart shows whether the selected customer profile is pushed toward higher or lower default risk.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================
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
