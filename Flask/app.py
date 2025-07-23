import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- Load the trained model and scaler ---
# The paths are relative to app.py. '..' means go up one directory to access model.pkl and scaler.pkl.
try:
    with open('../model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

try:
    with open('../scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None # Set scaler to None if loading fails

# Define the columns that the model expects as input features (11 features)
# This list MUST match the columns used in X_train from your notebook (excluding 'profile_id' and 'pm').
# The order here is crucial and must match the order in your X_train during model training.
input_features_columns = [
    'u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth', 'motor_speed',
    'i_d', 'i_q', 'stator_yoke', 'ambient', 'torque'
]

# --- Helper function for prediction logic ---
def make_prediction(form_data):
    """
    Processes form data, scales inputs, and makes a prediction using the loaded model.
    Returns the prediction text or an error message.
    """
    if model is None or scaler is None:
        return "Error: Model or scaler not loaded. Please check server logs."

    input_values = []
    # Iterate through the expected features to collect input values in the correct order
    for feature in input_features_columns:
        try:
            # Use .get() with a default value (e.g., 0.0) to handle cases where a field might be missing
            # or provide an error message if missing inputs are not allowed.
            # Using 0.0 as default for numerical inputs for robustness, but consider what makes sense for your data.
            input_values.append(float(form_data.get(feature, 0.0)))
        except ValueError:
            # If a value cannot be converted to float, return an error
            return f"Error: Invalid input for '{feature}'. Please enter numerical values."

    # Convert the list of input values to a NumPy array and reshape it.
    # .reshape(1, -1) converts it into a 2D array with 1 row and columns inferred automatically,
    # which is the format scikit-learn's scaler and model expect for a single prediction.
    input_array = np.array(input_values).reshape(1, -1)

    # Scale the input features using the loaded scaler. This is CRUCIAL.
    # The new input data MUST be scaled using the same scaler fitted on the training data.
    scaled_input = scaler.transform(input_array)

    # Make prediction using the loaded model
    # [0] is used because model.predict() returns an array, even for a single prediction.
    prediction = model.predict(scaled_input)[0]

    # Display prediction with units, as the target ('pm') was not normalized during training.
    # It represents actual temperature in Celsius.
    return f'Predicted Rotor Temperature: {prediction:.2f} °C'

# --- Define routes ---

@app.route('/')
def home():
    """Renders the home page with navigation options (Manual vs. Sensor input)."""
    return render_template('home.html')

@app.route('/manual_predict')
def manual_predict_page():
    """Renders the manual input form page."""
    return render_template('Manual_predict.html')

@app.route('/sensor_predict')
def sensor_predict_page():
    """Renders the sensor input form page."""
    return render_template('Sensor_predict.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    """Handles prediction requests from the manual input form (Manual_predict.html)."""
    prediction_text = make_prediction(request.form)
    # Render the same page, passing the prediction text back to display it
    return render_template('Manual_predict.html', prediction_text=prediction_text)

@app.route('/predict_sensor', methods=['POST'])
def predict_sensor():
    """Handles prediction requests from the sensor input form (Sensor_predict.html)."""
    prediction_text = make_prediction(request.form)
    # Render the same page, passing the prediction text back to display it
    return render_template('Sensor_predict.html', prediction_text=prediction_text)


# --- Run the Flask app ---
if __name__ == '__main__':
    # 'debug=True' is great for development as it automatically reloads the server
    # when you make changes to app.py and provides a debugger.
    # For a production environment, you would set debug=False.
    app.run(debug=True)
