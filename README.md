# K-Means Customer Clustering
 
A machine learning project that implements K-means clustering to segment mall customers into distinct groups based on their spending behavior and demographics.
 
## 📋 Project Overview
 
This project performs customer segmentation analysis using K-means clustering algorithm. It includes Jupyter notebooks for experimentation and a Streamlit web application for interactive visualization and prediction of customer clusters.
 
## ✨ Features
 
- **Customer Segmentation**: Clusters mall customers into groups using K-means algorithm
- **Interactive Web App**: Streamlit-based interface for exploring clusters and making predictions
- **Data Visualization**: Multiple visualizations including 2D/3D cluster plots and distributions
- **Model Persistence**: Pre-trained models saved with joblib for quick predictions
- **Data Processing**: Standardized features for optimal clustering performance
- **Exploratory Analysis**: Jupyter notebooks for detailed analysis and experimentation
 
## 📁 Project Structure
 
```
K-mean/
├── Streamlit_app.py                                    # Main Streamlit web application
├── K_mean_clustering.ipynb                             # Primary K-means clustering notebook
├── k-mean.ipynb                                        # Alternative clustering analysis notebook
├── Lab 1 - Classroom Exercise_KMeans_Clustering.ipynb  # Educational notebook
├── Mall_Customers.csv                                  # Original customer dataset
├── clustered_data.csv                                  # Processed data with cluster labels
├── clustered_mall_customers.csv                        # Final clustered customer data
├── test_fix.py                                         # Testing and debugging script
├── requirements.txt                                    # Python dependencies
└── README.md                                           # This file
```
 
## 🛠️ Installation
 
1. **Clone or download the project**
 
   ```bash
   cd "K-mean"
   ```
 
2. **Create a virtual environment (optional but recommended)**
 
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```
 
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
 
## 🚀 Usage
 
### Run the Streamlit Web App
 
```bash
streamlit run Streamlit_app.py
```
 
The app will open in your default browser at `http://localhost:8501`
 
### Run Jupyter Notebooks
 
```bash
jupyter notebook
```
 
Then select any of the `.ipynb` files to explore the clustering analysis step-by-step.
 
## 📊 Dataset
 
**Mall_Customers.csv** contains customer data with the following features:
 
- Customer ID
- Gender
- Age
- Annual Income
- Spending Score (1-100)
 
The data is preprocessed and standardized before applying K-means clustering.
 
## 📦 Dependencies
 
All required packages are listed in `requirements.txt`:
 
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library for K-means
- **joblib**: Model serialization
- **plotly**: Interactive visualizations
- **matplotlib & seaborn**: Static plotting libraries
 
## 🎯 How It Works
 
1. **Data Loading**: Import customer data from CSV
2. **Preprocessing**: Clean data and handle missing values
3. **Feature Scaling**: Standardize features using StandardScaler
4. **Clustering**: Apply K-means algorithm with optimal number of clusters
5. **Visualization**: Create interactive plots showing customer segments
6. **Prediction**: Predict cluster for new customer data
 
## 📝 Notes
 
- The K-means model is trained on the standardized features (Age, Annual Income, Spending Score)
- StandardScaler is used to normalize features to have mean 0 and standard deviation 1
- Pre-trained models are saved in the working directory for faster predictions
 
## 👨‍💻 Course Information
 
This is a student project from CVR College's Data Science/Machine Learning course on K-means clustering techniques.
 
## 📧 Support
 
For questions or issues, refer to the Jupyter notebooks which contain detailed explanations and comments.
 
---
 
**Last Updated**: February 2026
 
 
