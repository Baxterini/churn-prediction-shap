# 🏦 Bank Churn – Explainable AI (SHAP)

Interactive Streamlit application for customer churn prediction with local explanations using SHAP waterfall plots.

## 🔍 What does this app do?
- Predicts whether a bank customer is likely to churn
- Shows prediction probability
- Explains *why* the model made this decision (SHAP)
- Allows adjusting decision threshold
- Bilingual explanations (PL / EN)

## 🧠 Why Explainable AI?
Black-box models are not enough in banking.
This app demonstrates how ML predictions can be made transparent and interpretable.

## 📸 Screenshots
![Overview](screenshots/app_overview.png)
![SHAP](screenshots/shap_waterfall.png)

## 🚀 How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
