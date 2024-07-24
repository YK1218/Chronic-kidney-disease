from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the Random Forest model
with open('models/random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract features from form
        albumin = float(request.form['albumin'])
        sugar = float(request.form['sugar'])
        red_blood_cells = int(request.form['red_blood_cells'])
        serum_creatinine = float(request.form['serum_creatinine'])
        hemoglobin = float(request.form['hemoglobin'])
        red_blood_cell_count = float(request.form['red_blood_cell_count'])
        hypertension = int(request.form['hypertension'])
        diabetes_mellitus = int(request.form['diabetes_mellitus'])
        appetite = int(request.form['appetite'])
        pedal_edema = int(request.form['pedal_edema'])

        features = np.array([[albumin, sugar, red_blood_cells, serum_creatinine, hemoglobin,
                              red_blood_cell_count, hypertension, diabetes_mellitus, appetite,
                              pedal_edema]])

        # Predict using Random Forest
        prediction = random_forest_model.predict(features)
        
        if prediction[0] == 0:
            return redirect(url_for('result_good'))
        else:
            return redirect(url_for('result_bad'))

    return render_template('predict.html')

@app.route('/result_good')
def result_good():
    return render_template('result_good.html')

@app.route('/result_bad')
def result_bad():
    return render_template('result_bad.html')

if __name__ == '__main__':
    app.run(debug=True)
