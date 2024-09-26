from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model_path = '/mnt/data/model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    try:
        feature1 = float(request.form['Dependents'])
        feature2 = float(request.form['Self_Employed'])
        feature3 = float(request.form['ApplicantIncome'])
        feature4 = float(request.form['CoapplicantIncome'])
        feature5 = float(request.form['LoanAmount'])
        feature6 = float(request.form['Loan_Amount_Term'])
        feature7 = float(request.form['Credit_History'])
        # Add more features as needed based on the model input
    except ValueError:
        return "Invalid input. Please enter numeric values."

    # Create an array for the model
    features = np.array([[feature1, feature2, feature3,feature4,feature5,feature6,feature7]])  # Adjust based on the number of features
    prediction = model.predict(features)
    output = 'approved' if prediction[0] == 1 else 'REJECT'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))
if __name__ == '__main__':
    app.run(debug=True)
