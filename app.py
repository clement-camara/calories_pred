from flask import Flask, render_template, request
import numpy as np
import pickle
from flask_bootstrap import Bootstrap

#with open('model_pkl', 'rb') as f:
#    trained_model = pickle.load(f)

# load saved model
with open('model_pkl', 'rb') as f:
    model_pkl_trained_model = pickle.load(f)
    print(model_pkl_trained_model)

app = Flask(__name__)
# Flask-WTF requires an encryption key - the string can be anything
# app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'
# Flask-Bootstrap requires this line
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # get form data
        gender = request.form.get('gender')
        age = request.form.get('age')
        height = request.form.get('height')
        weight = request.form.get('weight')
        duration = request.form.get('duration')
        heart_rate = request.form.get('heart_rate')
        body_temp = request.form.get('body_temp')

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(gender, age, height, weight, duration, heart_rate, body_temp)
            # pass prediction to template
            return render_template('predict.html', prediction=prediction)

        except ValueError:
            return "Please Enter valid values"

    else:
        return "Please Enter valid values"


def preprocessDataAndPredict(gender, age, height, weight, duration, heart_rate, body_temp):
    # keep all inputs in array
    test_data = [gender, age, height, weight, duration, heart_rate, body_temp]
    print(test_data)

    # convert value data into numpy array
    test_data = np.array(test_data)

    # reshape array
    test_data = test_data.reshape(1, -1)
    print(test_data)
    prediction = model_pkl_trained_model.predict(test_data)
    print(prediction)
    return prediction


if __name__ == '__main__':
    app.run(debug=True)

