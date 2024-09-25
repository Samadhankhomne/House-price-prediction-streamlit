
import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r"C:\Users\91976\OneDrive\Desktop\GitHub\House Price Prediction\House_price.pkl", 'rb'))

# Set the title of the Streamlit app
st.title("House Price Prediction App")

# Add a brief description
st.write("This app predicts the House price based on SQF using a simple linear regression model.")

# Add input widget for user to enter square feet (SQF)
sqf = st.number_input("Enter SQF:", min_value=100.0, max_value=10000.0, value=100.0, step=50.0)

# When the button is clicked, make predictions
if st.button("Predict House price"):
    # Make a prediction using the trained model
    sqf_input = np.array([[sqf]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(sqf_input)
   
    # Display the result
    st.success(f"The predicted house price for {sqf} sqf is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of House price.")
