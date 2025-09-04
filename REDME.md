**Heart Disease Prediction ML Project**
📌 **Overview**

This project predicts the likelihood of heart disease using the UCI Heart Disease Dataset.
We apply a full machine learning pipeline: preprocessing, dimensionality reduction, feature selection, supervised & unsupervised models, hyperparameter tuning, and model deployment with Streamlit + Ngrok.

**📂 Project Structure**
Heart_Disease_Project/
│── data/
│   ├── heart_disease_clean.csv
│   ├── heart_disease_model_ready.csv
│   ├── heart_disease_selected_features.csv
│   ├── heart_disease_pca.csv
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_model_export.ipynb
│── models/
│   ├── final_model.pkl
│── results/
│   ├── evaluation_metrics.txt
│   ├── histograms_numeric.png
│   ├── correlation_heatmap.png
│   ├── target_distribution.png
│── ui/
│   ├── app.py  (Streamlit app)
│── deployment/
│   ├── ngrok_setup.txt
│── requirements.txt
│── README.md
│── .gitignore


**⚙️ Installation**
**Clone this repo:**
git clone https://github.com/<YourUsername>/Heart-Disease-ML-Project.git
cd Heart-Disease-ML-Project

**Create a virtual environment & install dependencies:**
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt

**📊 Dataset**
Source: UCI Heart Disease Dataset
Features: 13 predictors (age, sex, chest pain type, cholesterol, etc.)
Target: num → binary (0 = healthy, 1 = heart disease)
