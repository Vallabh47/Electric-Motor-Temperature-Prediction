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

# Define the columns that the model expects as input features
# This list MUST match the order and names of the features (X columns) used during model training in your notebook.
# These are all columns from the original dataset EXCEPT 'profile_id' and 'pm'.
input_features_columns = [
    'motor_speed', 'torque', 'i_d', 'i_q', 'u_d', 'u_q',
    'ambient', 'coolant', 'stator_winding', 'stator_tooth', 'stator_yoke',
    'psi_d', 'psi_q'
]


# --- Define routes ---

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the web form."""
    if model is None or scaler is None:
        return render_template('index.html', prediction_text="Error: Model or scaler not loaded. Please check server logs.")

    # Get data from the form
    data = request.form.to_dict()

    # Convert form data to a list of numerical values in the correct order
    # Ensure all required features are present and handle potential missing/invalid inputs
    input_values = []
    for feature in input_features_columns:
        try:
            # Use .get() with a default value (e.g., 0.0) to handle cases where a field might be missing
            # or provide an error message if missing inputs are not allowed.
            # Using 0.0 as default for numerical inputs for robustness, but consider what makes sense for your data.
            input_values.append(float(data.get(feature, 0.0)))
        except ValueError:
            return render_template('index.html', prediction_text=f"Error: Invalid input for '{feature}'. Please enter numerical values.")

    # Convert the list of input values to a NumPy array and reshape it.
    # .reshape(1, -1) converts it into a 2D array with 1 row and columns inferred automatically,
    # which is the format scikit-learn's scaler and model expect for a single prediction.
    input_array = np.array(input_values).reshape(1, -1)

    # Scale the input features using the loaded scaler. This is CRUCIAL.
    # The new input data MUST be scaled using the same scaler fitted on the training data.
    scaled_input = scaler.transform(input_array)

    # Make prediction using the loaded model
    prediction = model.predict(scaled_input)[0] # [0] because predict returns an array even for a single prediction

    # Display the result on the web page by passing it back to the template
    return render_template('index.html', prediction_text=f'Predicted Rotor Temperature: {prediction:.2f} °C')

# --- Run the Flask app ---
if __name__ == '__main__':
    # 'debug=True' is great for development as it automatically reloads the server
    # when you make changes to app.py and provides a debugger.
    # For a production environment, you would set debug=False.
    app.run(debug=True)