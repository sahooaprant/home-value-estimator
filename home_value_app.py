
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Sample dataset for Edison, NJ 08817
data = {
    'sqft': [2290, 1782, 2016, 2400, 1850],
    'bedrooms': [4, 3, 5, 4, 3],
    'bathrooms': [3, 1.5, 2, 3, 2],
    'zipcode': ['08817']*5,
    'price': [815000, 711000, 640000, 830000, 720000]
}
df = pd.DataFrame(data)

X = df[['sqft', 'bedrooms', 'bathrooms', 'zipcode']]
y = df['price']

categorical_features = ['zipcode']
numeric_features = ['sqft', 'bedrooms', 'bathrooms']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X, y)

st.title("Home Value Estimator for Edison, NJ (08817)")
sqft = st.number_input("Enter square footage:", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Enter number of bathrooms:", min_value=1, max_value=10, value=2)

if st.button("Estimate Price"):
    input_data = pd.DataFrame([[sqft, bedrooms, bathrooms, '08817']], columns=['sqft','bedrooms','bathrooms','zipcode'])
    estimated_price = model.predict(input_data)[0]
    st.write(f"Estimated Home Price: ${estimated_price:,.2f}")
