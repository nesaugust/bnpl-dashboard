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
except Exception:
    SHAP_AVAILABLE = False


st.set_page_config(
    page_title="BNPL Credit Risk Prediction Dashboard",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #f7faff;
    color: #061A33;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001B3F 0%, #002B5C 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #061A33;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 15px;
    color: #53657D;
    margin-bottom: 20px;
}

.card {
    background: white;
    border: 1px solid #E1E8F5;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0, 45, 100, 0.06);
}

.kpi-card {
    background: white;
    border: 1px solid #E1E8F5;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(0, 45, 100, 0.06);
}

.kpi-label {
    font-size: 13px;
    color: #64748B;
}

.kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #061A33;
}

.section-title {
    font-size: 20px;
    font-weight: 800;
    color: #071D49;
    margin-bottom: 12px;
}

.risk-low {
    background: #DDF8E8;
    color: #0B8F3C;
    padding: 14px;
    border-radius: 12px;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
}

.risk-medium {
    background: #FFF4CC;
    color: #B88400;
    padding: 14px;
    border-radius: 12px;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
}

.risk-high {
    background: #FFE1E1;
    color: #D62828;
    padding: 14px;
    border-radius: 12px;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
}

.recommend-box {
    background: #EAF8F0;
    color: #0F5132;
    padding: 16px;
    border-radius: 12px;
    font-size: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #005BFF, #287DFF);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-weight: 700;
    width: 100%;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E1E8F5;
    border-radius: 18px;
    padding: 18px;
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


def get_base_model(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model", model)
    return model


def get_probability(model, input_data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_data)[0][1]
    return float(model.predict(input_data)[0])


def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#071D49"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#0B63F6"},
            "bgcolor": "#EEF3FB",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#DDF8E8"},
                {"range": [30, 60], "color": "#FFF4CC"},
                {"range": [60, 100], "color": "#FFE1E1"},
            ],
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        font=dict(color="#071D49")
    )
    return fig


def get_feature_importance(model, feature_columns):
    base_model = get_base_model(model)

    if hasattr(base_model, "feature_importances_"):
        values = base_model.feature_importances_
    elif hasattr(base_model, "coef_"):
        values = np.abs(base_model.coef_[0])
    else:
        values = np.ones(len(feature_columns))

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": values
    }).sort_values("Importance", ascending=False).head(10)

    return importance_df


def calculate_local_explanation(model, input_data, feature_columns):
    base_model = get_base_model(model)

    try:
        if SHAP_AVAILABLE:
            if hasattr(base_model, "feature_importances_"):
                explainer = shap.TreeExplainer(base_model)
                shap_values = explainer.shap_values(input_data)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                values = shap_values[0]

            elif hasattr(base_model, "coef_"):
                values = base_model.coef_[0] * input_data.iloc[0].values

            else:
                values = np.zeros(len(feature_columns))
        else:
            importance = get_feature_importance(model, feature_columns)
            values = np.zeros(len(feature_columns))
            for feature in importance["Feature"]:
                idx = feature_columns.index(feature)
                values[idx] = importance.loc[importance["Feature"] == feature, "Importance"].iloc[0]

    except Exception:
        importance = get_feature_importance(model, feature_columns)
        values = np.zeros(len(feature_columns))
        for feature in importance["Feature"]:
            idx = feature_columns.index(feature)
            values[idx] = importance.loc[importance["Feature"] == feature, "Importance"].iloc[0]

    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": values,
        "Impact": np.abs(values)
    }).sort_values("Impact", ascending=False).head(10)

    return shap_df


try:
    model = load_model()
    feature_columns = load_feature_columns()
except Exception as e:
    st.error("App failed to load the model or feature columns.")
    st.exception(e)
    st.stop()

try:
    df = load_data()
except Exception:
    df = None
    st.warning("BNPL_dataset.csv was not found. Dashboard insights will be hidden.")


with st.sidebar:
    st.markdown("## BNPL RISK")
    st.markdown("---")
    st.markdown("### Dashboard")
    st.markdown("Prediction")
    st.markdown("- Predict Credit Risk")
    st.markdown("- Batch Prediction")
    st.markdown("Analytics")
    st.markdown("- Portfolio Overview")
    st.markdown("- Data Insights")
    st.markdown("Explainable AI")
    st.markdown("- Model Explainability")
    st.markdown("- SHAP Analysis")
    st.markdown("---")
    st.markdown("### Model Information")
    st.write("Model Type")
    st.write(type(get_base_model(model)).__name__)
    st.write("Trained On")
    st.write("BNPL Dataset")


st.markdown('<div class="main-title">BNPL Credit Risk Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict customer default risk using machine learning</div>', unsafe_allow_html=True)


if df is not None:
    total_customers = len(df)
    default_rate = df["default_flag"].mean() * 100
    avg_credit = df["credit_score"].mean()
    avg_income = df["monthly_income"].mean()
    high_risk_customers = int((df["default_flag"] == 1).sum())

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Customers</div><div class="kpi-value">{total_customers:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Default Rate</div><div class="kpi-value">{default_rate:.2f}%</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Credit Score</div><div class="kpi-value">{avg_credit:.0f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Income</div><div class="kpi-value">${avg_income:,.0f}</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk Customers</div><div class="kpi-value">{high_risk_customers:,}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


left, mid, right = st.columns([2.4, 1, 1.5])

with left:
    st.markdown('<div class="section-title">Predict Credit Risk</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        employment_type = st.selectbox("Employment Type", ["Employed", "Self-Employed", "Student", "Unemployed"])
        purchase_amount = st.number_input("Purchase Amount ($)", min_value=0.0, value=500.0)
        missed_payments = st.number_input("Missed Payments", min_value=0, value=0)
        location = st.selectbox("Location", ["Australia", "Canada", "Germany", "India", "UK", "USA"])

    with c2:
        monthly_income = st.number_input("Monthly Income ($)", min_value=0.0, value=5000.0)
        debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=5.0, value=0.30)
        bnpl_installments = st.selectbox("BNPL Installments", [1, 2, 3, 4, 6, 8, 10, 12], index=3)
        risk_score = st.slider("Risk Score", 0.0, 1.0, 0.50)
        customer_segment = st.selectbox("Customer Segment", ["High Risk", "Low Risk", "Medium Risk"])

    with c3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        app_usage_frequency = st.number_input("App Usage Frequency", min_value=0.0, value=10.0)
        repayment_delay_days = st.number_input("Repayment Delay Days", min_value=0, value=0)
        product_category = st.selectbox("Product Category", ["Beauty", "Electronics", "Fashion", "Home", "Sports"])
        transaction_date = st.date_input("Transaction Date")

    predict_button = st.button("Predict Risk")


transaction_month = transaction_date.month
transaction_year = transaction_date.year
transaction_day = transaction_date.day

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


try:
    probability = get_probability(model, input_data)
except Exception:
    probability = 0.0

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


with mid:
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            <p><b>Default Probability</b></p>
            <h1 style="color:#005BFF; text-align:center;">{probability * 100:.2f}%</h1>
            <hr>
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
    st.plotly_chart(create_gauge(probability), use_container_width=True)

    shap_df = calculate_local_explanation(model, input_data, feature_columns)
    top_factors = shap_df.sort_values("Impact", ascending=True).tail(5)

    fig_local = px.bar(
        top_factors,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        title="Risk Factors Contribution (Top 5)",
        color="SHAP Value",
        color_continuous_scale=["#1D6BFF", "#FF3B5C"]
    )
    fig_local.update_layout(
        height=280,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#061A33"),
        margin=dict(l=10, r=10, t=45, b=10)
    )
    st.plotly_chart(fig_local, use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True)

shap_col, force_col, imp_col = st.columns([1.3, 1.5, 1.3])

shap_df = calculate_local_explanation(model, input_data, feature_columns)
importance_df = get_feature_importance(model, feature_columns)

with shap_col:
    st.markdown('<div class="section-title">SHAP Summary - Feature Impact</div>', unsafe_allow_html=True)
    fig_shap = px.scatter(
        shap_df,
        x="SHAP Value",
        y="Feature",
        size="Impact",
        color="SHAP Value",
        color_continuous_scale=["#1D6BFF", "#FF3B5C"],
        title="Local SHAP Impact"
    )
    fig_shap.update_layout(
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#061A33"),
        margin=dict(l=10, r=10, t=45, b=10)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

with force_col:
    st.markdown('<div class="section-title">SHAP Force Plot - Local Explanation</div>', unsafe_allow_html=True)

    positive = shap_df[shap_df["SHAP Value"] > 0].head(5)
    negative = shap_df[shap_df["SHAP Value"] < 0].head(5)

    st.markdown(
        f"""
        <div class="card">
            <p>Impact of features for this prediction</p>
            <h2 style="text-align:center; color:#005BFF;">f(x) = {probability:.4f}</h2>
            <p style="text-align:center;">Base value: 0.50</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:
        st.write("Higher risk")
        st.dataframe(
            positive[["Feature", "SHAP Value"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

    with p2:
        st.write("Lower risk")
        st.dataframe(
            negative[["Feature", "SHAP Value"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True
        )

with imp_col:
    st.markdown('<div class="section-title">Feature Importance - Global</div>', unsafe_allow_html=True)
    fig_imp = px.bar(
        importance_df.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Mean Feature Impact",
        color_discrete_sequence=["#005BFF"]
    )
    fig_imp.update_layout(
        height=360,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#061A33"),
        margin=dict(l=10, r=10, t=45, b=10)
    )
    st.plotly_chart(fig_imp, use_container_width=True)


if df is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        score_group = df.copy()
        score_group["credit_score_group"] = pd.cut(
            score_group["credit_score"],
            bins=[300, 400, 500, 600, 700, 800, 850],
            labels=["300-400", "400-500", "500-600", "600-700", "700-800", "800-850"]
        )
        score_rate = score_group.groupby("credit_score_group", observed=False)["default_flag"].mean().reset_index()
        score_rate["default_rate"] = score_rate["default_flag"] * 100

        fig_score = px.line(
            score_rate,
            x="credit_score_group",
            y="default_rate",
            markers=True,
            title="Default Rate by Credit Score"
        )
        fig_score.update_layout(height=300, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_score, use_container_width=True)

    with d2:
        income_group = df.copy()
        income_group["income_group"] = pd.cut(
            income_group["monthly_income"],
            bins=[0, 2000, 4000, 6000, 8000, 10000, np.inf],
            labels=["< $2K", "$2K-$4K", "$4K-$6K", "$6K-$8K", "$8K-$10K", "> $10K"]
        )
        income_rate = income_group.groupby("income_group", observed=False)["default_flag"].mean().reset_index()
        income_rate["default_rate"] = income_rate["default_flag"] * 100

        fig_income = px.bar(
            income_rate,
            x="income_group",
            y="default_rate",
            title="Default Rate by Income Group",
            color_discrete_sequence=["#005BFF"]
        )
        fig_income.update_layout(height=300, paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_income, use_container_width=True)

    with d3:
        if "customer_segment" in df.columns:
            segment_df = df["customer_segment"].value_counts().reset_index()
            segment_df.columns = ["Customer Segment", "Count"]

            fig_segment = px.pie(
                segment_df,
                names="Customer Segment",
                values="Count",
                hole=0.55,
                title="Default Rate by Customer Segment"
            )
            fig_segment.update_layout(height=300, paper_bgcolor="white")
            st.plotly_chart(fig_segment, use_container_width=True)

    with d4:
        if "location" in df.columns:
            loc_rate = df.groupby("location")["default_flag"].mean().reset_index()
            loc_rate["default_rate"] = loc_rate["default_flag"] * 100
            loc_rate = loc_rate.sort_values("default_rate", ascending=False)

            fig_loc = px.bar(
                loc_rate,
                x="location",
                y="default_rate",
                title="Default Rate by Location",
                color_discrete_sequence=["#005BFF"]
            )
            fig_loc.update_layout(height=300, paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig_loc, use_container_width=True)


st.markdown(
    """
    <div style="
        background:#FFF8DC;
        border:1px solid #F2D98D;
        border-radius:12px;
        padding:16px;
        margin-top:20px;
        color:#5C4A00;
    ">
        Dashboard ini menggunakan model machine learning untuk memprediksi risiko gagal bayar pada transaksi BNPL.
        Hasil prediksi dan penjelasan SHAP digunakan sebagai alat bantu keputusan, bukan satu-satunya faktor penentu.
    </div>
    """,
    unsafe_allow_html=True
)