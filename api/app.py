from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the ML Model API"

@app.route('/hello', methods=['GET'])
def hello():
    name = request.args.get('name', 'World')
    return jsonify({"message": f"Hello, {name}!"})

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify, render_template
from predict_module import make_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        prediction = make_prediction(data['features'])
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict-form', methods=['GET', 'POST'])
def predict_form():
    if request.method == 'POST':
        study_hours = float(request.form['study_hours'])
        previous_scores = float(request.form['previous_scores'])
        prediction = make_prediction([study_hours, previous_scores])
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
# Task 2: Build a Flask Application
# 1)Create a basic Flask web app with the following:
# Routes:
# / – Home page with project info
# /predict – GET and POST endpoint for model prediction
from flask import Flask, request, jsonify
from predict_module import make_prediction

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Welcome to the Student Score Predictor API! Use POST /predict with features to get predictions."

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            "message": "Send a POST request with JSON body: { 'features': [study_hours, previous_score] }"
        })
    elif request.method == 'POST':
        try:
            data = request.get_json()
            features = data['features']
            prediction = make_prediction(features)
            return jsonify({'prediction': round(prediction, 2)})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


#Functionality:
#1)Accept user input through a web form or JSON
from flask import Flask, request, jsonify, render_template
from predict_module import make_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()  
        features = data['features']  
        prediction = make_prediction(features)
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        study_hours = float(request.form['study_hours'])
        previous_score = float(request.form['previous_score'])
        prediction = make_prediction([study_hours, previous_score])
        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)





#5)Return prediction result to the user
# a)For API (JSON response)
# @app.route('/predict', methods=['POST'])
# def predict_api():
#     try:
#         data = request.get_json()
#         features = data['features']  # e.g., [5, 80]
#         prediction = make_prediction(features)
#         return jsonify({'prediction': round(prediction, 2)})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400
# b)For Web Form (HTML page)
# This is for when the user submits a form in the browser:
# @app.route('/predict-form', methods=['POST'])
# def predict_form():
#     try:
#         study_hours = float(request.form['study_hours'])
#         previous_score = float(request.form['previous_score'])
#         prediction = make_prediction([study_hours, previous_score])
#         return render_template('index.html', prediction=round(prediction, 2))
#     except Exception as e:
#         return render_template('index.html', error=str(e))

# Optional Enhancements:
#1)Add error handling for invalid input
from flask import Flask, request, jsonify, render_template
from predict_module import make_prediction, get_metadata

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# JSON API prediction route
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()

        # Check if JSON body exists
        if not data:
            return jsonify({'error': 'No JSON payload found'}), 400

        # Check if 'features' key exists
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in JSON'}), 400

        features = data['features']

        # Validate that features is a list
        if not isinstance(features, list):
            return jsonify({'error': 'Features must be provided as a list'}), 400

        # Check the length of features (we expect 2: study_hours, previous_score)
        if len(features) != 2:
            return jsonify({'error': 'Features must contain exactly 2 values: [study_hours, previous_score]'}), 400

        # Convert features to floats and handle non-numeric input
        try:
            features = [float(x) for x in features]
        except ValueError:
            return jsonify({'error': 'All feature values must be numeric'}), 400

        # Make prediction
        prediction = make_prediction(features)

        return jsonify({
            'prediction': round(prediction, 2),
            'metadata': get_metadata()
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Form-based prediction endpoint
@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        study_hours = float(request.form['study_hours'])
        previous_score = float(request.form['previous_score'])
        prediction = make_prediction([study_hours, previous_score])
        return render_template('index.html', prediction=round(prediction, 2))
    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.")
    except Exception as e:
        return render_template('index.html', error=f"Unexpected error: {str(e)}")


@app.route('/metadata', methods=['GET'])
def metadata_api():
    return jsonify(get_metadata())

if __name__ == '__main__':
    app.run(debug=True)

#2)Include model metadata (accuracy, version)
from flask import Flask,request,jsonify,render_template
from predict_module import make_prediction,get_metadata

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400

        features = data["features"]
        if not isinstance(features, list) or len(features) != 2:
            return jsonify({'error': 'Features should be a list of [study_hours, previous_score]'}), 400

        prediction = make_prediction(features)
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
        previous_score = float(request.form['previous_score'])
        prediction = make_prediction([study_hours, previous_score])
        return render_template('index.html', prediction=round(prediction, 2))
    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.")
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/metadata', methods=['GET'])
def metadata_api():
    return jsonify(get_metadata())

if __name__ == '__main__':
    app.run(debug=True)

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
# 2)Submit data to /predict route via POST
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

# The prediction appears on the page.


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

# Mini Project: Student Performance Prediction API

#1)Accepts student features via a web form or API call
# a) Browser
# Start the server:

# python app.py
# Open a browser and go to:
# http://127.0.0.1:5000/

#i have Filled in Study Hours and Attendance, click Predict, and see the prediction result displayed on the page.

# b) Postman
# Create a POST request to:
# http://127.0.0.1:5000/predict

# In Body → raw → JSON, send:


# {
#   "features": [5, 75]
# }

# {
#   "prediction": 65.0,
#   "metadata": {
#     "model": "LinearRegression",
#     "version": "1.0",
#     "accuracy": "Demo data"
#   }
# }

#2)Returns a predicted performance score or grade
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from predict_module import predict_performance

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()

    try:
        features = {
            'hours_studied': float(data['hours_studied']),
            'attendance': float(data['attendance']),
            'previous_score': float(data['previous_score'])
        }

        score, grade = predict_performance(features)
        return jsonify({
            'score': score,
            'grade': grade,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


# 3)Runs locally and can be demonstrated via browser or Postman
# a) Run the Flask App
# In your terminal:
# python app.py
# The server will start at:

# http://127.0.0.1:5000/
# b) Browser Demo
# Open your browser and go to http://127.0.0.1:5000/.

# i have Filled out the form (e.g., Study Hours, Attendance).

# i had Submit the form.
# The prediction result will be displayed on the same page.

# c) Postman Demo
# Open Postman and create a POST request:


# http://127.0.0.1:5000/predict
# In Body → raw → JSON, enter:

# {
#   "features": [5, 80]
# }
# Click Send.

# You’ll get a JSON response like:


# {
#   "prediction": 68.5,
#   "metadata": {
#     "model": "LinearRegression",
#     "version": "1.0",
#     "accuracy": "Demo data"
#   }
# }






# Assignment Tasks
# Task 1: Prepare a Trained Model
# 1)Use a trained ML model from Week 6 (e.g., Student Score Predictor)

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

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

print("Model trained and saved as student_score_model.pkl")

# 2)save the model  using joblib or pickle
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

# 3)Ensure consistent preprocessing steps are reusable (e.g., scaling, encoding)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib


data = {
    'study_hours': np.random.uniform(1, 10, 50),
    'attendance': np.random.uniform(50, 100, 50),
    'assignments': np.random.randint(1, 10, 50)
}
df = pd.DataFrame(data)
df['final_score'] = (
    5 * df['study_hours'] +
    0.3 * df['attendance'] +
    2 * df['assignments'] +
    np.random.normal(0, 5, 50)
)


df.to_csv('students_performance.csv', index=False)
print("Dummy dataset 'students_performance.csv' created.")


X = df[['study_hours', 'attendance', 'assignments']]
y = df['final_score']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


joblib.dump(model, 'student_score_model.pkl')
joblib.dump(scaler, 'preprocessor.pkl')

print("Model and preprocessor saved successfully!")

#  Task 2: Build a Flask Application
# 1)Create a basic Flask web app with the following:
# Routes:
# / – Home page with project info
# /predict – GET and POST endpoint for model prediction
from flask import Flask, request, jsonify
from predict_module import make_prediction

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Welcome to the Student Score Predictor API! Use POST /predict with features to get predictions."

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            "message": "Send a POST request with JSON body: { 'features': [study_hours, previous_score] }"
        })
    elif request.method == 'POST':
        try:
            data = request.get_json()
            features = data['features']
            prediction = make_prediction(features)
            return jsonify({'prediction': round(prediction, 2)})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

#Functionality:
#1)Accept user input through a web form or JSON
from flask import Flask, request, jsonify, render_template
from predict_module import make_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()  
        features = data['features']  
        prediction = make_prediction(features)
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        study_hours = float(request.form['study_hours'])
        previous_score = float(request.form['previous_score'])
        prediction = make_prediction([study_hours, previous_score])
        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

# 2)Load the saved model
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

# 3)Perform preprocessing (if required)
import joblib
import numpy as np

model = joblib.load('student_score_model.pkl')  

def make_prediction(features):
    
    features_array = np.array(features).reshape(1, -1)
    

    
    prediction = model.predict(features_array)
    return prediction[0]

#4)Return prediction result to the user
# a)For API (JSON response)
# @app.route('/predict', methods=['POST'])
# def predict_api():
#     try:
#         data = request.get_json()
#         features = data['features']  # e.g., [5, 80]
#         prediction = make_prediction(features)
#         return jsonify({'prediction': round(prediction, 2)})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400
# b)For Web Form (HTML page)
# This is for when the user submits a form in the browser:
# @app.route('/predict-form', methods=['POST'])
# def predict_form():
#     try:
#         study_hours = float(request.form['study_hours'])
#         previous_score = float(request.form['previous_score'])
#         prediction = make_prediction([study_hours, previous_score])
#         return render_template('index.html', prediction=round(prediction, 2))
#     except Exception as e:
#         return render_template('index.html', error=str(e))



# Optional Enhancements:
#1)Add error handling for invalid input
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

       
        if not data:
            return jsonify({'error': 'No JSON payload found'}), 400

       
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in JSON'}), 400

        features = data['features']

       
        if not isinstance(features, list):
            return jsonify({'error': 'Features must be provided as a list'}), 400

       
        if len(features) != 2:
            return jsonify({'error': 'Features must contain exactly 2 values: [study_hours, previous_score]'}), 400


        try:
            features = [float(x) for x in features]
        except ValueError:
            return jsonify({'error': 'All feature values must be numeric'}), 400

        prediction = make_prediction(features)

        return jsonify({
            'prediction': round(prediction, 2),
            'metadata': get_metadata()
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        study_hours = float(request.form['study_hours'])
        previous_score = float(request.form['previous_score'])
        prediction = make_prediction([study_hours, previous_score])
        return render_template('index.html', prediction=round(prediction, 2))
    except ValueError:
        return render_template('index.html', error="Please enter valid numeric values.")
    except Exception as e:
        return render_template('index.html', error=f"Unexpected error: {str(e)}")


@app.route('/metadata', methods=['GET'])
def metadata_api():
    return jsonify(get_metadata())

if __name__ == '__main__':
    app.run(debug=True)



#2)Include model metadata (accuracy, version)
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


