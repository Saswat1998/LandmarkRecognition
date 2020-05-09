from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
#import numpy as np
import io
from PIL import Image
#import keras
from keras import backend as K
from keras.models import Sequential
#from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from keras.models import model_from_json

# Flask utils
from flask import redirect, url_for, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
def get_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global loaded_model
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model._make_predict_function()
    print(" * model loaded")
    return loaded_model
          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
loaded_model = get_model()
print('Model loaded. Check http://127.0.0.1:5000/')

def preprocess_image(image):
    im = Image.open(image)
    newsize = (128, 128)
    im = im.resize(newsize)
    imgSample = (np.array(im))
    imgSample = imgSample.reshape(-1,128,128,3)/255
    return imgSample


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(224, 224))
    preprocessedImg = preprocess_image(img_path)
    temp = loaded_model.predict(preprocessedImg)[0].max()
    val = np.where(loaded_model.predict(preprocessedImg)[0] == temp)[0][0]
    return str(val)
    

    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, loaded_model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

