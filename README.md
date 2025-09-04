**Heart Disease Prediction ML Project**

**📌 Overview**

This project predicts the likelihood of heart disease using the UCI Heart Disease Dataset.
We apply a full machine learning pipeline: preprocessing, dimensionality reduction, feature selection, supervised & unsupervised models, hyperparameter tuning, and model deployment with Github.

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

**🧪 Methodology**

1️⃣ Data Preprocessing

Handled missing values (median for numeric, mode for categorical).
Encoded categorical features (One-Hot Encoding).
Scaled numeric features (StandardScaler).


2️⃣ Dimensionality Reduction (PCA)

Applied PCA to reduce dimensionality.
Chose components covering 95% variance.
Visualized variance explained & scatter plots.


3️⃣ Feature Selection

Feature Importance (RandomForest, XGBoost).
Recursive Feature Elimination (RFE).
Chi-Square Test.
Selected optimal subset of predictors.


4️⃣ Supervised Learning

Trained models:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
Metrics: Accuracy, Precision, Recall, F1-score, AUC, ROC curves.


5️⃣ Unsupervised Learning

KMeans Clustering (Elbow method to choose K).
Hierarchical Clustering (Dendrogram analysis).
Compared clusters with true disease labels (ARI score).


6️⃣ Hyperparameter Tuning

GridSearchCV & RandomizedSearchCV.
Optimized Logistic Regression, Decision Tree, Random Forest, SVM.
Selected best performing model.


7️⃣ Model Export & Deployment
Saved final pipeline as .pkl with preprocessing + model.
Deployed via Streamlit web app.
Ngrok used for external access.

**📈 Results**

Best model: Random Forest (tuned)

Metrics (example — replace with your results):

Model	Accuracy	Precision	Recall	F1	AUC

Logistic Regression	0.84	0.85	0.80	0.83	0.90

Decision Tree	0.82	0.81	0.79	0.80	0.85

Random Forest	0.87	0.88	0.85	0.86	0.92

SVM	0.85	0.86	0.82	0.84	0.89

📌 See results/evaluation_metrics.txt for full metrics.
