import numpy as np
import pickle
from flask import Flask, request, render_template

#create flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open('rfc3.pkl','rb'))


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/route', methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features  = [np.asarray(float_features)]
    prediction = model.predict(features)

    return render_template('index.html',prediction_text="the employee is {}".format(prediction))

if __name__ == 'main':
    app.run(debug=True)
