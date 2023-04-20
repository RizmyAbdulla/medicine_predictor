from flask import Flask, request, jsonify
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# load the model
model = load_model('model.h5')

# load the dataset
data = pd.read_csv('dataset.csv')

disease = pd.get_dummies(data['Disease'])

with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():

    symptoms_to_predict = request.json['symptoms']

    symptoms_array = mlb.transform([symptoms_to_predict])

    disease_prediction = model.predict(np.expand_dims(symptoms_array, axis=2))

    predicted_disease_index = np.argmax(disease_prediction)

    predicted_disease = disease.columns[predicted_disease_index]

    predicted_antibiotics = data.loc[data['Disease'] == predicted_disease, 'Antibiotics'].values[0]

    output = {
        'disease': predicted_disease,
        'antibiotics': predicted_antibiotics
    }

    # return the output as JSON
    response = jsonify(output)
    return response

if __name__ == '__main__':
    app.run(debug=False)
