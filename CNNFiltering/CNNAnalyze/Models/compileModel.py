import json
import tensorflow as tf

f=open('model/layer_map_model.json','r')
model = tf.keras.models.model_from_json(json.load(f))
model.load_weights('model/layer_map_model_last.h5')
f.close()

# https://gist.github.com/carlthome/6ae8a570e21069c60708017e3f96c9fd
import tf2xla_pb2

config = tf2xla_pb2.Config()
batch_size = 1

for x in model.inputs:
    x.set_shape([batch_size] + list(x.shape)[1:])
    feed = config.feed.add()
    feed.id.node_name =x.op.name
    feed.shape.MergeFrom(x.shape.as_proto())

for x in model.outputs:
    fetch = config.fetch.add()
    fetch.id.node_name = "Identity"

with open('graph.config.pbtxt', 'w') as f:
    f.write(str(config))

import shutil

shutil.copyfile("graph.pb","tensorflow-r1.15/graph.pb")
shutil.copyfile("graph.config.pbtxt","tensorflow-r1.15/graph.config.pbtxt")
shutil.copyfile("BUILD","tensorflow-r1.15/BUILD")

# bazel build --show_progress_rate_limit=600 @org_tensorflow//:graph
# in tensorflow-r1.15/

# tensorflow-r1.15/bazel-bin/external/org_tensorflow/graph.h