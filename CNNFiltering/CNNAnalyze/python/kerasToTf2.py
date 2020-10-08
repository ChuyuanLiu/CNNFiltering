import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
import json

K.set_learning_phase(0)

f=open('model/layer_map_model.json','r')
keras_model = model_from_json(json.load(f))
keras_model.load_weights('model/layer_map_model_last.h5')
f.close()

keras_model.save("model/layer_map_model")