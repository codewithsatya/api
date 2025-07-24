# Assignment Tasks
# Task 1: Prepare a Trained Model
# 1)Use a trained ML model from Week 6 (e.g., Student Score Predictor)

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import pickle

# data = {
#     'study_hours': [1,2,3,4,5,6,7,8,9,10],
#     'previous_scores': [40,42,45,50,55,58,60,65,70,75],
#     'final_score': [42,45,50,55,60,65,68,72,78,82]
# }
# df = pd.DataFrame(data)

# X = df[['study_hours', 'previous_scores']]
# y = df['final_score']

# model = LinearRegression()
# model.fit(X, y)

# with open('student_score_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("Model trained and saved as student_score_model.pkl")



# Optional Enhancements:
# 1)Add error handling for invalid input
# import pickle
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# x=np.array([[1,40],[2,50],[3,60],[4,70],[5,75],[6,80]])
# y=np.array([45,50,55,65,75,85])
# model=LinearRegression()
# model.fit(x,y)
# y_pred=model.predict(x)
# accuracy=r2_score(y,y_pred)
# model_data={
#        "model": model,
#     "metadata": {
#         "version": "1.0",
#         "accuracy": round(accuracy, 3),
#         "features": ["study_hours", "previous_score"]
# }
# }
# with open("student_model.pkl", "wb") as f:
#     pickle.dump(model_data, f)

# print("Model trained and saved as student_model.pkl with metadata")
# Task 3: Design a Simple HTML Form (Frontend)
# Create a basic HTML form for user input (e.g., hours studied, attendance, etc.)
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import joblib


# data = pd.DataFrame({
#     'study_hours': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'attendance': [60, 65, 70, 75, 80, 85, 90, 92, 95],
#     'final_score': [50, 55, 60, 65, 70, 75, 80, 85, 90]
# })

# X = data[['study_hours', 'attendance']]
# y = data['final_score']

# model = LinearRegression()
# model.fit(X, y)


# joblib.dump(model, 'student_model.pkl')
# print("Model trained and saved as student_model.pkl")

# import pickle
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score


# X = np.array([
#     [2, 60],
#     [3, 65],
#     [4, 70],
#     [5, 75],
#     [6, 80]
# ])
# y = np.array([50, 55, 60, 65, 70])

# model = LinearRegression()
# model.fit(X, y)


# metadata = {
#     "accuracy": r2_score(y, model.predict(X)),
#     "version": "1.0",
#     "features": ["study_hours", "attendance"]
# }


# with open("iris_model.pkl", "wb") as f:
#     pickle.dump({"model": model, "metadata": metadata}, f)

# print("Model trained and saved successfully.")

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import pickle


# data = {
#     'hours_studied': [2, 3, 4, 5, 6, 7, 8],
#     'attendance': [60, 65, 70, 75, 80, 85, 90],
#     'previous_score': [50, 55, 60, 65, 70, 75, 80],
#     'performance_score': [55, 60, 65, 70, 75, 80, 85]
# }

# df = pd.DataFrame(data)


# X = df[['hours_studied', 'attendance', 'previous_score']]
# y = df['performance_score']


# model = LinearRegression()
# model.fit(X, y)


# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("Model trained and saved as model.pkl")

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



 Mini Project: Student Performance Prediction API

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

