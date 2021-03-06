import tensorflow as tf
import json
import keras2onnx
import onnx
import shutil
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

#https://github.com/onnx/keras-onnx
def KerasToONNX(keras_model):
    onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name, target_opset=9)
    keras2onnx.save_model(onnx_model,"model/layer_map_model.onnx")

def KerasToTF2(keras_model):
    keras_model.save("model/layer_map_model")

def loadTest():
    onnx_model = onnx.load("model/layer_map_model.onnx")
    print(onnx_model)
    print(onnx_model.opset_import)

def loadModel():
    f=open('model/layer_map_model.json','r')
    keras_model = tf.keras.models.model_from_json(json.load(f))
    keras_model.load_weights('model/layer_map_model_last.h5')
    f.close()
    print(keras_model.summary())
    return keras_model

model=loadModel()
print(model.summary())