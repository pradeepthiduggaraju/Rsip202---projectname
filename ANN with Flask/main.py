# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:45:26 2019

@author: Sai Nidhi
"""
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
global model,graph
import tensorflow as tf
# load the pre-trained Keras model

model = load_model('colabregressor.h5')
graph = tf.get_default_graph()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/age')
@app.route('/', methods=['GET','POST'])
def form_post():
    a = request.form['rd']
    b= request.form['ad']
    c = request.form['ms']
    d = request.form['s']
    if (d == "us"):
        s1,s2,s3 = 0,0,1
    if (d == "america"):
        s1,s2,s3 = 0,0,1
    else:
        s1,s2,s3 = 0,0,1
        
    total = [[s1,s2,s3,a,b,c]]
    print(total)
    with graph.as_default():
        y_pred = model.predict(np.array(total))
    return render_template('base.html',ypred = str(y_pred[0][0]))
    
if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=False)
    
    

    

        

        
    