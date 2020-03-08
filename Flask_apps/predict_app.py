import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from keras.models import model_from_json

app=Flask(__name__)

def get_model():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	global loaded_model
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	print(" * model loaded")

def preprocess_image(image):
	im = Image.open(image)
	newsize = (128, 128)
	im = im.resize(newsize)
	imgSample = (np.array(im))
	imgSample = imgSample.reshape(-1,128,128,3)/255
	return imgSample

print(" Loading model ")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	#print(decoded)
	image = Image.open(io.BytesIO(decoded))
	processed_image = preprocess_image(image)
	temp = loaded_model.predict(processed_image)[0].max()
	val = np.where(loaded_model.predict(processed_image)[0] == temp)[0][0]

	response = {
		'prediction': {
			'landmark_id': val,
			'landmark_name': "St.Petersburgh Church"
		}
	}
	return jsonify(response)

