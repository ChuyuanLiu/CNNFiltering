import json
import tensorflow as tf
import shutil
import os

modelName='model/layer_map_model'

f=open(modelName + '.json','r')
model = tf.keras.models.model_from_json(json.load(f))
model.load_weights(modelName + '_last.h5')
f.close()

shutil.copyfile("tensorflow/tensorflow/compiler/tf2xla/tf2xla_pb2.py","tf2xla_pb2.py")

# https://gist.github.com/carlthome/6ae8a570e21069c60708017e3f96c9fd
import tf2xla_pb2

config = tf2xla_pb2.Config()
batch_size = 500

for x in model.inputs:
    x.set_shape([batch_size] + list(x.shape)[1:])
    feed = config.feed.add()
    feed.id.node_name =x.op.name
    feed.shape.MergeFrom(x.shape.as_proto())

for x in model.outputs:
    fetch = config.fetch.add()
    fetch.id.node_name = x.op.name

with open('graph.config.pbtxt', 'w') as f:
    f.write(str(config))
tf.keras.backend.clear_session()

os.remove("tf2xla_pb2.py")

dest = "config/"
#dest = "tensorflow/"
src = "tfAOT/"
import saveModelTf1
saveModelTf1.savePb(modelName)
shutil.copyfile(src+"BUILD",dest+"BUILD")
shutil.copyfile(src+"tfAOT.cc",dest+"tfAOT.cc")
shutil.copyfile(src+"tfAOT.h",dest+"tfAOT.h")
shutil.move("graph.pb",dest+"graph.pb")
shutil.move("graph.config.pbtxt",dest+"graph.config.pbtxt")

# os.chdir(dest)
# os.system("bazel build --show_progress_rate_limit=600 libtfaot.so")
# os.chdir("../")
# shutil.copyfile(dest+"bazel-bin/libtfaot.so","tfAOT/libtfaot.so")

#### compile test.cc
# g++ -o test test.cc tfAOT.h libtfaot.so
# export LD_LIBRARY_PATH=~/data/Tracking/CNNFiltering/CNNFiltering/CNNAnalyze/Models/tfAOT
# ./test

#### requirement
# CentOS 7 (match SLC 7)
# bazel 0.26.1 (binary https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-linux-x86_64)
# gcc 8.2.X (match the version used in CMSSW)
# git 2.X
# tensorflow r1.15 source code (https://github.com/tensorflow/tensorflow/archive/r1.15.zip)
