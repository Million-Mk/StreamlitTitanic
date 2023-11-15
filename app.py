import streamlit as st
from joblib import load

# Hello world StreamLit
# st.title('Hello world StreamLit')

# load the model from disk
model = load('titanic_model.joblib')

# Create StreamLit web app
st.title('Titanic Survival Parameters')

# Sidebar with bar
st.sidebar.title('Home')

# Menu option
menu = ['Home', 'Predition']

# Input with sliders
st.sidebar.selectbox('', menu)

age = st.slider('Age', 0.42, 80.0, 30.0)
sibsp = st.slider('Sibsp', 0, 8, 0)
parch = st.slider('Parch', 0, 9, 0)
fare = st.slider('Fare', 0.0, 512.30, 32.20)

# Add button
predict_button = st.button('Predict')

if predict_button:
    # รับค่ามาเก็บในตัวแปรแบบ list
    input_data = [[age, sibsp, parch, fare]]

    # ทำนายผล
    prediction = 3
    prediction = model.predict(input_data)
    # st.write(prediction[0])

    # แสดงผลลัพย์
    st.subheader('prediction')
    if (prediction[0] == 1):
        st.write('Survived')
    else:
        st.write('Not Survived')

    # หาค่าความน่าจะเป็น
    predict_proba = model.predict_proba(input_data)

    # แสดงค่าความน่าจะเป็น
    st.subheader('Prediction Probability')
    st.write(f'Survived {predict_proba[0][1]:2f}')
    st.write(f'Not Survived: {predict_proba[0][0]:2f}')
