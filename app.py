import streamlit as st 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#---------------------- streamlit app ----------------------

st.title("Wine Quality Prediction App")

# Loading the Dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

st.image('wine quality.jpeg')  #st.image is used to display images in streamlit

st.caption("Wine Quality Dataset Explorer and RandomForest Classifier Model")

# subheader

st.subheader("Dataset",divider="violet")
st.write(data.head())

# EDA subheader

st.subheader("Exploratory Data Analysis (EDA)" , divider="violet")

# buttons for the EDA

if st.button("Column names"):
    st.write("Dataset columns",data.columns)
    
if st.button("Missing values"):
    st.write("Sum of missing values in each column",data.isnull().sum())


st.subheader("Data visualization",divider="violet")

#data viz checkbox
if st.checkbox("Bar plot of residual sugar against quality"):
     st.bar_chart(x="residual sugar", y="quality",data=data)
     
if st.checkbox("line plot of residual sugar against quality"):
     st.line_chart(x="residual sugar", y="quality",data=data)
     

#preapare the data

X = data.drop("quality",axis = 1)
y = data["quality"]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=42)


#create and fit our model

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)


#User input

st.sidebar.header("Slide your values")

fixed_acidity = st.sidebar.slider("Fixed Acidity", (data['fixed acidity']).min(),(data['fixed acidity']).max(),(data['fixed acidity']).mean())

volatile_acidity = st.sidebar.slider("Volatile Acidity", (data['volatile acidity'].min()), (data['volatile acidity'].max()), (data['volatile acidity'].mean()))

citric_acid = st.sidebar.slider("Citric Acid", (data['citric acid'].min()), (data['citric acid'].max()), (data['citric acid'].mean()))

residual_sugar = st.sidebar.slider("Residual Sugar", (data['residual sugar'].min()), (data['residual sugar'].max()), (data['residual sugar'].mean()))

chlorides = st.sidebar.slider("Chlorides", (data['chlorides'].min()), (data['chlorides'].max()), (data['chlorides'].mean()))

free_suplhur_dioxide = st.sidebar.slider("free suplhur dioxide", (data['free sulfur dioxide'].min()), (data['free sulfur dioxide'].max()), (data['free sulfur dioxide'].mean()))

total_sulfur_dioxide =  st.sidebar.slider("total sulfur dioxide", (data['total sulfur dioxide'].min()), (data['total sulfur dioxide'].max()), (data['total sulfur dioxide'].mean()))

density=st.sidebar.slider("density", (data['density'].min()), (data['density'].max()), (data['density'].mean()))

pH=st.sidebar.slider("pH", (data['pH'].min()), (data['pH'].max()), (data['pH'].mean()))

sulphates=st.sidebar.slider("sulphates", (data['sulphates'].min()), (data['sulphates'].max()), (data['sulphates'].mean()))

alcohol=st.sidebar.slider("alcohol", (data['alcohol'].min()), (data['alcohol'].max()), (data['alcohol'].mean()))



#predict button

if st.sidebar.button("Predict"):
    
    #create datafarme for the user inputs
    
    user_input = pd.DataFrame(
        {
           'fixed acidity':[fixed_acidity],
           'volatile acidity':[volatile_acidity],
           'citric acid':[citric_acid],
           'residual sugar':[residual_sugar],
           'chlorides' :[chlorides],
           'free sulfur dioxide':[free_suplhur_dioxide],
           'total sulfur dioxide':[total_sulfur_dioxide],
           'density':[density],
           'pH':[pH],
           'sulphates':[sulphates],
           'alcohol':[alcohol]   
        }
    )
    
    #prediction of quality of wine
    
    prediction = rf.predict(user_input)
    
    #display of the prediction
    st.sidebar.subheader('prediction')
    st.sidebar.write(f"From the information provied the wine quality is {prediction[0]}")