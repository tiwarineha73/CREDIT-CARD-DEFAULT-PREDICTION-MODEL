# 💳 CreditGuard AI — Credit Card Default Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3d4f7c?style=flat-square&logo=plotly)

A **production-level ML web application** that predicts credit card default risk using a Random Forest model trained on 30,000 real-world records from the UCI Credit Card Default dataset.

## 🌐 Live Features

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, dataset stats, model KPIs |
| 🎯 Prediction | Interactive form → risk score + gauge + recommendations |
| 📊 Data Analysis | 10+ interactive Plotly charts (EDA) |
| 🧠 Model Insights | Confusion matrix, feature importance, architecture |
| 📥 Report Download | Formatted text report download |

## 🚀 Run Locally

```bash
git clone https://github.com/tiwarineha73/<repo-name>
cd credit_default_app
pip install -r requirements.txt
streamlit run app.py
```

## 📁 File Structure

```
credit_default_app/
├── app.py              # Main Streamlit application
├── model.joblib        # Trained Random Forest model
├── scaler.joblib       # StandardScaler (used in training)
├── metrics.json        # Model performance metrics
├── data.csv            # UCI Credit Card Default dataset
├── requirements.txt    # Dependencies
└── README.md
```

## 🤖 Model Details

- **Algorithm**: Random Forest Classifier
- **Trees**: 200 estimators, max depth 10
- **Class weight**: Balanced (handles 22% default imbalance)
- **Features**: 23 (payment status, bill amounts, demographics)
- **Train/Test**: 80/20 stratified split

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 78.8% |
| Precision | 52.0% |
| Recall | 56.1% |
| F1 Score | 54.0% |
| ROC-AUC | 77.3% |

## 🗂️ Dataset

- **Source**: UCI Machine Learning Repository
- **Records**: 30,000 Taiwan credit card clients
- **Target**: Default payment next month (binary)
- **Features**: Credit limit, payment history (6 months), bill amounts, demographics

## 👩‍💻 Built By

**Neha Tiwari** — [github.com/tiwarineha73](https://github.com/tiwarineha73)
