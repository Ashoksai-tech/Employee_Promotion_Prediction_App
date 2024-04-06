# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 00:45:55 2024

@author: aasho
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:14:08 2024
@author: aasho
"""

import pickle
import numpy as np

# Load the saved model
loaded_model = pickle.load(open("C:/Users/aasho/OneDrive/Desktop/employee_model/logistic.pkl",'rb'))

def promotion_pred(input_data):
    input_data_as_array = np.array(input_data)
    
    # Reshape the input data
    reshape_data = input_data_as_array.reshape(1, -1)
    
    prediction = loaded_model.predict(reshape_data)
    
    if prediction[0] == 0:
        return 'Not Promoted'
    else:
        return 'Promoted'
    
def main():
    import streamlit as st

    st.title('Employee Promotion Predictor')

    # Input fields with more descriptive labels
    department = st.selectbox('Department', ['Select Department', 'Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D', 'Procurement', 'Finance', 'HR'])
    region = st.selectbox('Region', ['Select Region', 'region_1', 'region_2', 'region_3', 'region_4', 'region_5', 'region_6', 'region_7', 'region_8', 'region_9'])
    education = st.selectbox('Education', ['Select Education', 'Master\'s & above', 'Bachelor\'s', 'Below Secondary'])
    gender = st.selectbox('Gender', ['Select Gender', 'Male', 'Female'])
    recruitment_channel = st.selectbox('Recruitment Channel', ['Select Recruitment Channel', 'sourcing', 'referred', 'other'])
    no_of_trainings = st.number_input('Number of Trainings', min_value=0)
    age = st.number_input('Age', min_value=0)
    previous_year_rating = st.selectbox('Previous Year Rating', ['Select Rating', 1.0, 2.0, 3.0, 4.0, 5.0])
    length_of_service = st.number_input('Length of Service', min_value=0)
    kpi_met = st.selectbox('KPIs Met >80%', ['Select Option', 0, 1])
    awards_won = st.selectbox('Awards Won?', ['Select Option', 0, 1])
    avg_training_score = st.number_input('Average Training Score', min_value=0)
    
    # Button to trigger prediction
    promotion_prediction = ''
    if st.button('Predict Promotion'):
        promotion_prediction = promotion_pred([department, region, education, gender, recruitment_channel,
                     no_of_trainings, age, previous_year_rating, length_of_service,
                     kpi_met, awards_won, avg_training_score])
      
        st.success(promotion_prediction)
      
if __name__ == '__main__':
    main()
