# BNPL Credit Risk Dashboard

A machine learning-powered dashboard for predicting **Buy Now Pay Later (BNPL) credit risk** using customer financial and behavioural data.

This project combines **Exploratory Data Analysis (EDA)**, **Machine Learning**, and an interactive **Streamlit dashboard** to help analyse repayment risk and customer profiles.

---

## Live Demo

https://bnpl-dashboard-quk85wuz7werjwk8jkkjsm.streamlit.app/

---

## GitHub Repository

https://github.com/nesaugust/bnpl-dashboard

---

# Project Overview

This project was developed to analyse customer repayment behaviour in BNPL services and predict the likelihood of credit default.

The workflow includes:

- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model development
- Model evaluation & selection
- Interactive dashboard deployment

The final dashboard allows users to:

- Predict customer credit risk
- Analyse financial behaviour
- Explore repayment trends
- Visualise important risk indicators
- Understand feature importance using SHAP-style explanations

---

# Dataset Features

The dataset contains customer demographic, financial, and repayment-related information such as:

- Age
- Monthly Income
- Credit Score
- Purchase Amount
- BNPL Installments
- Repayment Delay Days
- Employment Type
- Product Category
- Customer Segment
- Previous Defaults
- Spending Behaviour

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# Machine Learning Workflow

## 1. Data Preprocessing

The preprocessing stage included:

- Handling missing values
- One-hot encoding categorical variables
- Feature engineering from transaction dates
- Converting boolean variables
- Removing unnecessary ID columns

The machine learning pipeline was built using Scikit-learn and XGBoost. :contentReference[oaicite:0]{index=0}

---

## 2. Exploratory Data Analysis (EDA)

EDA was conducted to identify:

- Customer repayment patterns
- Income distribution
- Default behaviour
- Correlation between financial variables
- High-risk customer segments

Visualisations include:

- Distribution plots
- Correlation heatmaps
- Risk segmentation charts
- Feature importance analysis

---

## 3. Model Development

Several machine learning models were tested and compared, including:

- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost
- Neural Network

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The best-performing model was selected automatically based on ROC-AUC score and saved as:

```bash
best_bnpl_model.pkl
```

The pipeline also saves:

```bash
feature_columns.json
model_results.csv
```

---

# Dashboard Features

The Streamlit dashboard includes:

- Credit risk prediction
- Interactive customer input filters
- Risk probability scoring
- Global feature importance
- SHAP-style explanation charts
- Financial analytics visualisations
- Responsive modern UI design

---

# Project Structure

```bash
bnpl-dashboard/
│
├── app.py
├── best_bnpl_model.pkl
├── feature_columns.json
├── model_results.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── Eda_Ml Model.ipynb
│
├── models/
│   └── final_model_selection.py
│
└── dataset/
    └── BNPL_dataset.csv
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/nesaugust/bnpl-dashboard.git
```

Move into the project folder:

```bash
cd bnpl-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

---

# Dashboard Preview

Add screenshots of your dashboard here.

Example:

```bash
images/dashboard_preview.png
```

---

# Future Improvements

Potential future enhancements include:

- Real-time API integration
- Advanced explainable AI visualisations
- Fraud detection module
- Customer segmentation clustering
- Cloud deployment optimisation
- Time-series repayment forecasting

---

# Learning Outcomes

Through this project, I gained practical experience in:

- End-to-end machine learning workflow
- Financial risk analytics
- Exploratory data analysis
- Feature engineering
- Model evaluation and comparison
- Dashboard development using Streamlit
- Explainable AI concepts
- Machine learning deployment

---

# Author

Agnes Jeni  

GitHub: https://github.com/nesaugust
