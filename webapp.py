#Description:  This app defects if a persaon has Diabetes using Machine Learning and Python

#Importing required libraries
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Create a title and subtitle
st.write("""
# Diabetes Detection 
Detect if someone has diabetes using ML and Python
""")

# model_image = Image("")

#Get The data for Diabetes detection

df = pd.read_csv("C:/Users/Vishal Kumar/PycharmProjects/DiabetesDetectionModel/diabetes.csv")

#set a subheader
st.subheader("Data Information")

#Show the data as table
st.dataframe(df)

# Show statistics on the data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into Independent 'X' and dependent 'y'
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

# Split the dataset into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Get the feature input from the user
def get_user_input():

    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction
    pregnancies = st.sidebar.slider("pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("glucose", 0, 199, 117)
    blood_pressure = st.sidebar.slider("blood_pressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("skin_thickness", 0, 99, 23)
    insulin = st.sidebar.slider("insulin", 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    DiabetesPedigreeFunction = st.sidebar.slider("DiabetesPedigreeFunction", 0.078, 2.42, 0.3725)
    age = st.sidebar.slider("age", 21, 81, 30)

    # Store a dictionary into a variable
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'age': age
    }

    # To automate above inputs
    # def get_user_input():
    #     user_data = {}
    #     for feature in df.columns[:-1]:
    #         slider_max = (df[feature].max() * 2)
    #         if 'int' in str(type(df[feature][0])):
    #             data = st.sidebar.slider(str(feature), 0, slider_max, 0)
    #         else:
    #             data = st.sidebar.slider(str(feature), 0.0, slider_max, 0.0)
    #         user_data[feature] = data
    #     features = pd.DataFrame(user_data, index=[0])
    #     return features

    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user_input into a variable
user_input = get_user_input()

# Set a subheader and display the users input
st.subheader("User Input: ")
st.write(user_input)

# Create and train the model
RandomForstClassifier = RandomForestClassifier()
RandomForstClassifier.fit(X_train, Y_train)

# Show the model's metrics
st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(Y_test, RandomForstClassifier.predict(X_test)) * 100) + '%')

# Store the model's predictions in a variable
prediction = RandomForstClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)