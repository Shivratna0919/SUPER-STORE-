# 🛒 BigMart Sales Prediction System

A Machine Learning project that predicts **BigMart outlet sales** using a trained **K-Nearest Neighbors (KNN) Regression** model with an interactive Streamlit dashboard.

## ✨ Features

- 🛒 Item & outlet sales prediction
- 🤖 KNN Regression model
- 📊 35 processed model features
- ⚡ Instant sales prediction
- 🎨 Modern Streamlit dashboard
- 📋 Processed input preview

## 🧠 ML Workflow

```text
BigMart Dataset
      ↓
Data Preprocessing
      ↓
Categorical Encoding
      ↓
MinMaxScaler
      ↓
KNN Regression
      ↓
Outlet Sales Prediction
      ↓
Streamlit Dashboard
```

## 🤖 Model

- **Algorithm:** K-Nearest Neighbors (KNN) Regression
- **Neighbors:** 7
- **Distance:** Minkowski
- **p:** 2
- **Scaler:** MinMaxScaler
- **Features:** 35 processed features
- **Target:** Item Outlet Sales

## 🗂️ Project Structure

```text
BigMart-Sales-Prediction/
│
├── app.py
├── bigmart_knn_model.pkl
├── KNN Regression.ipynb
├── KNN_reg_outlet_sales - KNN_reg_outlet_sales.csv
├── requirements.txt
└── README.md
```

## ⚙️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- KNN Regression

## 🚀 Installation & Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📌 Input Features

The application uses product and outlet information such as:

- Item Weight
- Item Fat Content
- Item Visibility
- Item Type
- Item MRP
- Outlet Identifier
- Outlet Size
- Outlet Location
- Outlet Type
- Outlet Establishment Year

## 🎯 Purpose

This project demonstrates an end-to-end Machine Learning workflow for **retail sales prediction** with a user-friendly web interface.

**For educational and project demonstration purposes.**
