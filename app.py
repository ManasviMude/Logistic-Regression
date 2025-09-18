
# 6. DEPLOYMENT WITH STREAMLIT
# --- train_model.py ---

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("diabetes.csv")
cols_zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols_zero_missing] = df[cols_zero_missing].replace(0, np.nan)

X = df.drop(columns=['Outcome'])
y = df['Outcome']

numeric_features = X.columns.tolist()
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features)])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "logreg_pipeline.pkl")
print("Saved model")
"""

# --- app.py ---
"""
import streamlit as st
import pandas as pd
import joblib

st.title("Diabetes Risk Predictor")

model = joblib.load("logreg_pipeline.pkl")

preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
bp = st.number_input("BloodPressure", min_value=0.0, value=70.0)
skin = st.number_input("SkinThickness", min_value=0.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, value=79.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, value=0.47)
age = st.number_input("Age", min_value=0, max_value=120, value=33)

input_df = pd.DataFrame([{
    'Pregnancies': preg,
    'Glucose': glucose,
    'BloodPressure': bp,
    'SkinThickness': skin,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age
}])

if st.button("Predict"):
    proba = model.predict_proba(input_df)[0,1]
    pred = model.predict(input_df)[0]
    st.write(f"Predicted probability of diabetes: **{proba:.3f}**")
    st.write("Prediction:", "**Diabetic**" if pred==1 else "**Non-diabetic**")

