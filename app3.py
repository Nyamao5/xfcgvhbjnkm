import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Streamlit app
st.title("Wine Quality Prediction App")

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')


st.image("wine quality.jpeg")
# Streamlit app
st.caption("Wine Quality Dataset Explorer and RandomForest Classifier Model")


# Display the dataset
st.subheader("Dataset",divider="violet")
st.write(data.head())


# EDA subheader
st.subheader("Exploratory Data Analysis (EDA)",divider="violet")


#Exploratory data analysis
 
if st.button("Column Names"):
    st.write("Dataset Column Names",data.columns)


# Check for missing values
if st.button("Missing Values"):
    st.write("Number of missing values in each column:",data.isnull().sum())



st.subheader("Data Visualization",divider="violet")
# Data visualization
if st.checkbox("Bar Plot of residual sugar against quality"):
    st.bar_chart(x="residual sugar",y='quality',data=data)
    
if st.checkbox("Line Plot of residual sugar against quality"):
    st.line_chart(x="residual sugar",y='quality',data=data)    


    

# Prepare the data
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)



# User input for features
st.sidebar.header("Slide Your Values")

fixed_acidity = st.sidebar.slider("Fixed Acidity", float(data['fixed acidity'].min()), float(data['fixed acidity'].max()), float(data['fixed acidity'].mean()))
volatile_acidity = st.sidebar.slider("Volatile Acidity", float(data['volatile acidity'].min()), float(data['volatile acidity'].max()), float(data['volatile acidity'].mean()))
citric_acid = st.sidebar.slider("Citric Acid", float(data['citric acid'].min()), float(data['citric acid'].max()), float(data['citric acid'].mean()))
residual_sugar = st.sidebar.slider("Residual Sugar", float(data['residual sugar'].min()), float(data['residual sugar'].max()), float(data['residual sugar'].mean()))
chlorides = st.sidebar.slider("Chlorides", float(data['chlorides'].min()), float(data['chlorides'].max()), float(data['chlorides'].mean()))
free_suplhur_dioxide = st.sidebar.slider("free suplhur dioxide", float(data['free sulfur dioxide'].min()), float(data['free sulfur dioxide'].max()), float(data['free sulfur dioxide'].mean()))
total_sulfur_dioxide =  st.sidebar.slider("total sulfur dioxide", float(data['total sulfur dioxide'].min()), float(data['total sulfur dioxide'].max()), float(data['total sulfur dioxide'].mean()))
density=st.sidebar.slider("density", float(data['density'].min()), float(data['density'].max()), float(data['density'].mean()))
pH=st.sidebar.slider("pH", float(data['pH'].min()), float(data['pH'].max()), float(data['pH'].mean()))
sulphates=st.sidebar.slider("sulphates", float(data['sulphates'].min()), float(data['sulphates'].max()), float(data['sulphates'].mean()))
alcohol=st.sidebar.slider("alcohol", float(data['alcohol'].min()), float(data['alcohol'].max()), float(data['alcohol'].mean()))


# Predict button
if st.sidebar.button("Predict"):
    # Create a DataFrame with user inputs
    user_input = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_suplhur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]})

    # Make prediction
    prediction = rf.predict(user_input)

    # Display the prediction
    st.sidebar.subheader("Prediction")
    st.sidebar.write(f"From the Information you provided the  predicted wine quality is {prediction[0]}")

