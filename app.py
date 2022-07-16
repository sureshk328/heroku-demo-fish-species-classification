import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    to_predict_list = request.form.to_dict()
    print(list(to_predict_list.values()))
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    final_features = [np.array(to_predict_list)]
    #prediction = model.predict(final_features)
    #{'Bream':0, 'Roach':1, 'Whitefish':2, 'Parkki':3, 'Perch':4, 'Pike':5, 'Smelt':6}
    result = model.predict(final_features)
    if int(result)== 0:
        prediction ='Bream'
    elif int(result)==1:
        prediction ='Roach'
    elif int(result)==2:
        prediction ='Whitefish'
    elif int(result)==3:
        prediction ='Parkki'
    elif int(result)==4:
        prediction ='Perch'
    elif int(result)==5:
        prediction ='Pike'
    elif int(result)==6:
        prediction ='Smelt'
    else:
        prediction="Unable to predict"



    return render_template('index.html', prediction_text='Predicted Fish Species {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
