import tensorflow as tf
import json
import keras2onnx
import onnx

#https://github.com/onnx/keras-onnx
def KerasToONNX():
    f=open('model/layer_map_model.json','r')
    keras_model = tf.keras.models.model_from_json(json.load(f))
    keras_model.load_weights('model/layer_map_model_last.h5')
    f.close()

    onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name, target_opset=9)
    keras2onnx.save_model(onnx_model,"model/layer_map_model.onnx")

def KerasToTF2():
    f=open('model/layer_map_model.json','r')
    keras_model = tf.keras.models.model_from_json(json.load(f))
    keras_model.load_weights('model/layer_map_model_last.h5')
    f.close()

    keras_model.save("model/layer_map_model")

def LoadTest():
    onnx_model = onnx.load("model/layer_map_model.onnx")
    print(onnx_model)
    print(onnx_model.opset_import)
    
KerasToONNX()