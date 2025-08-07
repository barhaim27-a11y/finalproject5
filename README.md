
# 🧠 Parkinson's Disease Prediction - Final ML & AI Project

This project is a complete machine learning pipeline and interactive web app that predicts the likelihood of Parkinson's disease based on vocal measurements.

---

## 📂 Project Structure

- `EDA_and_Modeling_Parkinsons_FINAL_GRIDSEARCH_SAVED.ipynb`  
  Full analysis notebook: EDA, preprocessing, model comparison, and model export.

- `app_styled_fixed.py`  
  Streamlit web application for real-time prediction with beautiful layout and visualization.

- `model.pkl`  
  Trained and optimized model saved with GridSearchCV using RandomForest + StandardScaler (Pipeline).

- `requirements.txt`  
  All Python packages required to run the app.

---

## 🚀 How to Run the App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app_styled_fixed.py
   ```

---

## 📊 Dataset

- Source: [UCI Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- Features: 22 vocal measurements per patient
- Target: `status` (1 = Parkinson's, 0 = Healthy)

---

## ✅ Model Details

- Algorithm: RandomForestClassifier
- Feature Scaling: StandardScaler
- Optimized via GridSearchCV (cross-validation)
- Model saved as `Pipeline` including scaler and classifier

---

## ✨ Features

- Full Exploratory Data Analysis (EDA)
- Multiple model training and comparison
- GridSearch hyperparameter tuning
- Interactive Streamlit app for predictions
- Visualizations of prediction results

---

Created with ❤️ as part of a final project for the ML & AI course.
