from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # your form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
    float(request.form['Age']),
    float(request.form['Duration of alcohol consumption(years)']),
    float(request.form['Quantity of alcohol consumption (quarters/day)']),
    float(request.form['TCH']),
    float(request.form['TG']),
    float(request.form['LDL']),
    float(request.form['HDL']),
    float(request.form['Hemoglobin  (g/dl)']),
    float(request.form['SGOT/AST      (U/L)']),
    float(request.form['SGPT/ALT (U/L)'])
]



        print("\n📥 Input received:", input_features)

        prediction = model.predict([input_features])[0]
        prediction_label = 'Likely' if prediction == 1 else 'Not Likely'

        print("📤 Model output:", prediction)
        print("🧠 Final label:", prediction_label)

        return render_template('result.html', prediction_text=f'Patient is {prediction_label} to have liver cirrhosis.')

    except Exception as e:
        print("❌ Prediction error:", e)
        return "Error in prediction: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)
