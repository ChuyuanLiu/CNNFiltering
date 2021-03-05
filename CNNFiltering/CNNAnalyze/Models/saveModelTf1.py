# tensorflow r1.15
# https://github.com/pipidog/keras_to_tensorflow/blob/master/keras_to_tensorflow.py
import tensorflow as tf
import json
from tensorflow.keras import backend as K

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,output_names, freeze_var_names)
        return frozen_graph

def loadModel():
    wkdir = '.'
    pb_filename = 'graph.pb'
    from tensorflow.python.platform import gfile
    with tf.Session() as sess:
        # load model from pb file
        with gfile.FastGFile(wkdir+'/'+pb_filename,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)
        # write to tensorboard (check tensorboard for each op names)
        writer = tf.summary.FileWriter(wkdir+'/log/')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()
        # print all operation names 
        print('\n===== ouptut operation names =====\n')
        for op in sess.graph.get_operations():
            print(op.name)
        
def savePb(modelName):
    K.set_learning_phase(0)
    f=open(modelName + '.json','r')
    model = tf.keras.models.model_from_json(json.load(f))
    model.load_weights(modelName + '_last.h5')
    f.close()
    frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, ".", "graph.pb", as_text=False)
    K.clear_session()