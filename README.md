# K-Means Customer Clustering

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)

A machine learning project that implements **K-means clustering** to segment mall customers into distinct groups based on their spending behavior and demographics.

---

## 📋 Project Overview

This project performs customer segmentation analysis using the **K-means clustering algorithm**. It includes:

- Jupyter notebook for clustering analysis (`Mall_Customers.ipynb`)  
- A Streamlit web application (`Streamlit_app.py`) for interactive visualization and prediction of customer clusters  
- Pre-trained models for quick predictions

---

## ✨ Features

- **Customer Segmentation:** Groups mall customers using K-means clustering  
- **Interactive Web App:** Streamlit interface for exploring clusters and predicting new customer segments  
- **Data Visualization:** 2D/3D cluster plots and distribution charts  
- **Model Persistence:** Pre-trained K-means and Random Forest models saved with `joblib` for fast predictions  
- **Data Processing:** Standardized features for optimal clustering performance  

---

## 📁 Project Structure


k-means-clustering/
├── data/
│ ├── Mall_Customers.csv # Original dataset
│ └── clustered_data_saved.csv # Processed dataset with cluster labels
├── notebooks/
│ └── Mall_Customers.ipynb # Clustering analysis notebook
├── models/
│ ├── kmeans_model.pkl # Pre-trained K-means model
│ ├── random_forest_model.pkl # Pre-trained Random Forest model
│ └── scaler.pkl # StandardScaler for feature normalization
├── Streamlit_app.py # Main Streamlit web application
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/praharshithaKasoju/k-means-clustering.git
cd k-means-clustering

Create a virtual environment (optional but recommended)

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt
🚀 Usage
Run the Streamlit Web App
streamlit run Streamlit_app.py

The app will open in your default browser at http://localhost:8501.

Run Jupyter Notebook
jupyter notebook

Open notebooks/Mall_Customers.ipynb to explore clustering analysis step by step.

📊 Dataset

Mall_Customers.csv contains customer data with these features:

Customer ID

Gender

Age

Annual Income

Spending Score (1-100)

The dataset is preprocessed and standardized before applying K-means clustering.

📦 Dependencies

Web app: streamlit

Data manipulation: pandas, numpy

Machine learning: scikit-learn

Model persistence: joblib

Visualizations: matplotlib, seaborn, plotly

🎯 How It Works

Data Loading: Import customer data from CSV

Preprocessing: Clean data and handle missing values

Feature Scaling: Standardize features using StandardScaler

Clustering: Apply K-means with optimal number of clusters

Visualization: Create interactive plots to explore customer segments

Prediction: Predict cluster for new customer data using pre-trained models

📝 Notes

K-means is trained on standardized features: Age, Annual Income, Spending Score

StandardScaler normalizes features to mean 0 and standard deviation 1

Pre-trained models are saved in models/ for faster predictions

👨‍💻 Course Information

This student project is part of CVR College's Data Science/Machine Learning course on K-means clustering techniques.

📧 Support

For questions or issues, refer to the Jupyter notebook, which contains detailed explanations and comments.

Last Updated: February 2026
