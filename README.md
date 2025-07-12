# King County Home Price Estimator

## Live app link
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://king-county-housing-app-2n4oafk8vjtwv5atjktmed.streamlit.app/)

## About

An interactive app that estimates house prices in King County, WA using machine learning.  
The model was trained on real housing data (Kaggle dataset, updated in 2016) and deployed with Streamlit for easy access.

- Built entirely in Google Colab
- Uses XGBoost with StandardScaler preprocessing
- Inputs include home size, location, condition, and more

## Project Highlights

- End-to-end ML pipeline: cleaning → modeling → evaluation → deployment
- Compared multiple models: Linear Regression, Random Forest, and XGBoost
- Final model (XGBoost) performance:
  - RMSE ≈ **148,000**
  - R² ≈ **0.86**

## Tech Stack

- Python (pandas, numpy, scikit-learn, xgboost, joblib)
- Streamlit
- Google Colab
- GitHub

---

