
import streamlit as st
import pandas as pd
import joblib

# load model scaler feature cols
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("King County Home Value Estimator")
st.caption("Built with XGBoost and Streamlit")

bedrooms = st.number_input("Bedrooms", 0, 10, 3)
bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0)
floors = st.number_input("Floors", 1.0, 3.5, 1.0)

# size
sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 1600)
sqft_lot = st.number_input("Lot Size (sqft)", 500, 100000, 5000)
sqft_above = st.number_input("Above Ground Sqft", 300, 10000, 1500)
sqft_basement = st.number_input("Basement Sqft", 0, 5000, 0)

# age/reno
yr_built = st.number_input("Year Built", 1900, 2022, 1985)
yr_renovated = st.number_input("Year Renovated", 0, 2022, 0)

# location
lat = st.number_input("Latitude", 47.0, 48.0, 47.5)
long = st.number_input("Longitude", -123.5, -121.0, -122.2)
zipcode = st.number_input("Zipcode", 98000, 98200, 98115)

# neighborhood avg
sqft_living15 = st.number_input("Living Area (Neighbor Avg)", 300, 10000, 1600)
sqft_lot15 = st.number_input("Lot Size (Neighbor Avg)", 500, 100000, 6000)

# quality
view = st.slider("View Score", 0, 4, 0)
condition = st.slider("Condition (1–5)", 1, 5, 3)
grade = st.slider("Grade (1–13)", 1, 13, 7)
waterfront = st.selectbox("Waterfront", [0, 1])  # 0 = no, 1 = yes

# format row for prediction
features = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                          condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
                          lat, long, sqft_living15, sqft_lot15, zipcode]],
                        columns=feature_names)

# scale inputs + predict
features_scaled = scaler.transform(features)

if st.button("Estimate Price"):
    prediction = model.predict(features_scaled)[0]
    st.success(f"Estimated Price: ${prediction:,.0f}")
