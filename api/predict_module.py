import pickle

with open('student_score_model.pkl', 'rb') as f:
    model = pickle.load(f)

def make_prediction(features):
    return model.predict([features])[0]

task-1
2)save the model  using joblib or pickle
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd


data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'previous_scores': [40,42,45,50,55,58,60,65,70,75],
    'final_score': [42,45,50,55,60,65,68,72,78,82]
}
df = pd.DataFrame(data)


X = df[['study_hours', 'previous_scores']]
y = df['final_score']
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'student_score_model.joblib')

print("Model saved as student_score_model.joblib")
# b)Using pickle 
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd


data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'previous_scores': [40,42,45,50,55,58,60,65,70,75],
    'final_score': [42,45,50,55,60,65,68,72,78,82]
}
df = pd.DataFrame(data)


X = df[['study_hours', 'previous_scores']]
y = df['final_score']
model = LinearRegression()
model.fit(X, y)


with open('student_score_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as student_score_model.pkl")

# Student Score Predictor with Scaling
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Sample data
data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'previous_scores': [40,42,45,50,55,58,60,65,70,75],
    'final_score': [42,45,50,55,60,65,68,72,78,82]
}
df = pd.DataFrame(data)

X = df[['study_hours', 'previous_scores']]
y = df['final_score']

# Create pipeline: Scaling + Linear Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Standardize features
    ('model', LinearRegression())         # Train model
])

# Train pipeline
pipeline.fit(X, y)

# Save pipeline with pickle
with open('student_score_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Pipeline with preprocessing saved as student_score_model.pkl")


# Prediction Code
import pickle

with open('student_score_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

features = [6, 65]  
prediction = pipeline.predict([features])

print(f"Predicted Final Score: {prediction[0]:.2f}")

import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), 'student_score_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def make_prediction(features):
    """Takes a list of features [study_hours, previous_score] and returns prediction."""
    prediction = model.predict([features])
    return prediction[0]

# Task 2: Build a Flask Application
# 3)Load the saved model
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), 'student_score_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)


def make_prediction(features):
    prediction = model.predict([features])
    return prediction[0]

import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'student_score_model.pkl')
model = joblib.load(model_path)

def make_prediction(features):
    return model.predict([features])[0]


# Task 2: Build a Flask Application
# 4)Perform preprocessing (if required)
import joblib
import numpy as np

model = joblib.load('student_score_model.pkl')  

def make_prediction(features):
    
    features_array = np.array(features).reshape(1, -1)
    

    
    prediction = model.predict(features_array)
    return prediction[0]

# Task 2: Build a Flask Application
# Optional Enhancements:
# 1)Add error handling for invalid input
import joblib

model = joblib.load("student_model.pkl")

metadata = {
    "model_name": "Student Score Predictor",
    "version": "1.0",
    "accuracy": 0.92,
    "features": ["study_hours", "previous_score"]
}

def make_prediction(features):
    return model.predict([features])[0]

def get_metadata():
    return metadata


# 2)Include model metadata (accuracy, version)
import joblib

model = joblib.load("student_model.pkl")


metadata = {
    "model_name": "Student Score Predictor",
    "version": "1.0",
    "accuracy": 0.92,
    "features": ["study_hours", "previous_score"]
}

def make_prediction(features):
    """Make prediction from input features."""
    prediction = model.predict([features])
    return prediction[0]

def get_metadata():
    """Return model metadata."""
    return metadata

# Create a basic HTML form for user input (e.g., hours studied, attendance, etc.)
import joblib
import numpy as np

# Load model
model = joblib.load('student_model.pkl')

def make_prediction(features):
    """ features = [study_hours, attendance] """
    features_array = np.array(features).reshape(1, -1)
    return model.predict(features_array)[0]

def get_metadata():
    return {
        "model": "Linear Regression",
        "version": "1.0",
        "accuracy": "Demo Data - No Validation"
    }

import pickle


with open("iris_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
metadata = data["metadata"]

def make_prediction(features):
    return model.predict([features])[0]

def get_metadata():
    return metadata

import pickle
import numpy as np


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_performance(features):
    X = np.array([[features['hours_studied'], features['attendance'], features['previous_score']]])
    score = model.predict(X)[0]
    score = round(score, 2)
    

    if score >= 80:
        grade = 'A'
    elif score >= 60:
        grade = 'B'
    else:
        grade = 'C'

    return score, grade


import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_performance(features):
    X = np.array([[features['hours_studied'], features['attendance'], features['previous_score']]])
    score = model.predict(X)[0]
    score = round(score, 2)

    if score >= 80:
        grade = 'A'
    elif score >= 60:
        grade = 'B'
    else:
        grade = 'C'

    return score, grade




# Task 3: Design a Simple HTML Form (Frontend)
#1)Create a basic HTML form for user input (e.g., hours studied, attendance, etc.)
from flask import Flask, request, jsonify, render_template
from predict_module import make_prediction, get_metadata

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400

        prediction = make_prediction(data['features'])
        return jsonify({
            'prediction': round(prediction, 2),
            'metadata': get_metadata()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        prediction = make_prediction([study_hours, attendance])
        return render_template('index.html', prediction=round(prediction, 2))
    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.")
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Student Score Predictor</title>
# </head>
# <body>
#     <h2>Predict Your Exam Score</h2>
#     <form action="/predict-form" method="POST">
#         <label for="study_hours">Study Hours:</label>
#         <input type="number" step="0.1" name="study_hours" required><br><br>

#         <label for="attendance">Attendance (%):</label>
#         <input type="number" step="1" name="attendance" required><br><br>

#         <button type="submit">Predict</button>
#     </form>

#     {% if prediction %}
#         <h3>Predicted Score: {{ prediction }}</h3>
#     {% endif %}

#     {% if error %}
#         <p style="color:red;">Error: {{ error }}</p>
#     {% endif %}
# </body>
# </html>



#2)Submit data to /predict route via POST
# i have used
# 1. Using JSON with /predict (API style)
# Endpoint: POST /predict

# Send JSON like:


# {
#   "features": [5, 80]
# }
# Using curl (Command Line):


# curl -X POST http://127.0.0.1:5000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"features":[5,80]}'
# Using Postman:

# Select POST method.

# URL: http://127.0.0.1:5000/predict

# Body → raw → JSON:


# {
#   "features": [5, 80]
# }
# Expected Response:

# {
#   "prediction": 65.0,
#   "metadata": {
#     "model": "Linear Regression",
#     "version": "1.0",
#     "accuracy": "Demo Data - No Validation"
#   }
# }
# 2. Using the Web Form
# Your HTML form submits data via POST to /predict-form.

# Fill in Study Hours and Attendance.

# Click Predict.

# The prediction appears on the page

#3)Display prediction result on the page
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("student_model.pkl")

# Metadata for your model
MODEL_METADATA = {
    "model_name": "Student Score Predictor",
    "version": "1.0",
    "accuracy": 0.92  # replace with your actual accuracy
}

# Prediction function
def make_prediction(features):
    prediction = model.predict([features])
    return prediction[0]

def get_metadata():
    return MODEL_METADATA

@app.route('/')
def home():
    return render_template('index.html')

# API endpoint for JSON requests
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400

        features = data["features"]
        if not isinstance(features, list) or len(features) != 2:
            return jsonify({'error': 'Features should be [study_hours, attendance]'}), 400

        prediction = make_prediction(features)
        return jsonify({
            'prediction': round(prediction, 2),
            'metadata': get_metadata()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Form submission route
@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        prediction = make_prediction([study_hours, attendance])
        return render_template('index.html', prediction=round(prediction, 2))
    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.")
    except Exception as e:
        return render_template('index.html', error=str(e))

# Metadata endpoint
@app.route('/metadata', methods=['GET'])
def metadata_api():
    return jsonify(get_metadata())

if __name__ == '__main__':
    app.run(debug=True)


# Optional:
# 1)Add Bootstrap or Tailwind CSS for styling
from flask import Flask, request, jsonify, render_template
from predict_module import make_prediction, get_metadata

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Initially no prediction or error

@app.route('/predict', methods=['POST'])
def predict_api():
    """JSON API endpoint for prediction"""
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400

        prediction = make_prediction(data['features'])
        return jsonify({
            'prediction': round(prediction, 2),
            'metadata': get_metadata()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-form', methods=['POST'])
def predict_form():
    """Form submission handler"""
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        prediction = make_prediction([study_hours, attendance])
        return render_template('index.html', prediction=round(prediction, 2))
    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.")
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)



# <!-- 2)Add a reset or "Try Again" button -->
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Student Score Predictor</title>
#     <script src="https://cdn.tailwindcss.com"></script>
# </head>
# <body class="bg-gray-100 min-h-screen flex items-center justify-center">

#     <div class="bg-white shadow-lg rounded-lg p-8 w-full max-w-md">
#         <h1 class="text-3xl font-bold mb-6 text-center text-blue-600">Student Score Predictor</h1>

#         <!-- Form -->
#         <form action="/predict-form" method="POST" class="space-y-4">
#             <div>
#                 <label for="study_hours" class="block mb-1 font-medium text-gray-700">Study Hours:</label>
#                 <input type="number" step="0.1" name="study_hours" id="study_hours" required
#                        class="border border-gray-300 rounded-lg w-full p-2 focus:ring-2 focus:ring-blue-500">
#             </div>

#             <div>
#                 <label for="attendance" class="block mb-1 font-medium text-gray-700">Attendance (%):</label>
#                 <input type="number" step="0.1" name="attendance" id="attendance" required
#                        class="border border-gray-300 rounded-lg w-full p-2 focus:ring-2 focus:ring-blue-500">
#             </div>

#             <div class="flex justify-between">
#                 <button type="submit"
#                         class="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600">
#                     Predict
#                 </button>
#                 <button type="reset"
#                         class="bg-gray-400 text-white px-4 py-2 rounded-lg hover:bg-gray-500">
#                     Reset
#                 </button>
#             </div>
#         </form>

#         <!-- Prediction Result -->
#         {% if prediction %}
#             <div class="mt-6 text-center">
#                 <p class="text-green-600 text-xl font-semibold">Predicted Score: {{ prediction }}</p>
#                 <a href="/" class="text-blue-500 underline mt-2 inline-block">Try Again</a>
#             </div>
#         {% endif %}

#         <!-- Error Message -->
#         {% if error %}
#             <div class="mt-6 text-center">
#                 <p class="text-red-600 font-semibold">{{ error }}</p>
#                 <a href="/" class="text-blue-500 underline mt-2 inline-block">Try Again</a>
#             </div>
#         {% endif %}
#     </div>

# </body>
# </html>


# Task 4: API Testing
# Test /predict using:
#1)Browser form submission
# Open your app in a browser:

# http://127.0.0.1:5000/
# Fill in Study Hours and Attendance.

# Click Predict.

# The predicted score should appear on the same page.

# Expected Input:

# study_hours: 5
# attendance: 80
# Expected Output (example):

# Predicted Score: 75.3

#2)Postman (send raw JSON payload)
# Open Postman.

# Create a new POST request to:


# http://127.0.0.1:5000/predict
# Go to Body → raw → select JSON.

# Send JSON payload:


# {
#   "features": [5, 80]
# }

# {
#   "prediction": 75.3,
#   "metadata": {
#       "accuracy": 0.92,
#       "version": "1.0"
#   }
# }

#3)curl (optional for CLI users)
# Run this command in Git Bash or Command Prompt:

#  -X POST http://127.0.0.1:5000/predict \
#      -H "Content-Type: application/json" \
#      -d "{\"features\": [5, 80]}"
# Expected Output:


# {"prediction": 75.3, "metadata": {"accuracy": 0.92, "version": "1.0"}}


#4)Document input/output formats in a README.md or markdown cell
# Example section for your README:


# ## API Documentation

# ### Endpoint: `/predict` (POST)
# - **Description**: Predicts a student's score based on study hours and attendance.

# #### Input Format (JSON)
# ```json
# {
#   "features": [study_hours, attendance]
# }
