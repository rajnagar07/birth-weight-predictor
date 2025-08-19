from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Function to clean and prepare data from the form
def get_clean_data(form_data):
    # Extract values from form and convert them to appropriate data types
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])

    # Store cleaned data in dictionary format
    cleaned_data = {
        "gestation": [gestation],
        "parity": [parity],
        "age": [age],
        "height": [height],
        "weight": [weight],
        "smoke": [smoke]
    }
    return cleaned_data

# Home route: renders the HTML form page
@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

# Prediction route: handles POST request when user submits the form
@app.route("/predict", methods=['POST']) 
def get_prediction():
    # Get raw form data submitted by user
    baby_data_form = request.form
    
    # Clean and structure the form data
    baby_data = get_clean_data(baby_data_form)
    
    # Convert dictionary to DataFrame for model input
    # Note: Must pass as list/dict with [ ] or transpose depending on model training
    baby_df = pd.DataFrame(baby_data)

    # Load the trained machine learning model from pickle file
    with open("model1.pkl", 'rb') as obj:
        mymodel = pickle.load(obj)

    # Make prediction on user input data
    prediction = mymodel.predict(baby_df)
    
    # Round prediction to 2 decimal places
    prediction = round(float(prediction), 2)

    # Return prediction 
    # response = {
    #     'prediction': prediction
    # }
    return render_template('index.html',prediction = prediction)



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
