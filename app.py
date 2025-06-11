from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('student.pkl')  # Make sure model.pkl exists in the same directory

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    stream = int(request.form.get('stream'))
    internships = int(request.form['internships'])
    cgpa = float(request.form['cgpa'])
    backlog = int(request.form['backlog'])

    # Format input for prediction
    features = np.array([[stream, internships, cgpa, backlog]])

    # Make prediction
    prediction = model.predict(features)[0]

    # Format result text
    prediction_text = "🎉 ✅ Student is likely to be Placed!" if prediction == 1 else "❌ Student is not likely to be Placed."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
