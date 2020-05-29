#---------------------------------------------------------------------------------------------------------------------
# script to convert tf to tvm
#
# please download files in https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/ folder to ./data
# the generated file will be saved to ./deploy
# ---------------------------------------------------------------------------------------------------------------------
import tvm
from tvm import te
from tvm import relay
import time
from tvm.contrib import util, graph_runtime as runtime


# os and numpy
import numpy as np
import os.path
import tvm
from tvm import te
import numpy as np
# Tensorflow imports
import tensorflow as tf


try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing



# Base location for model related files.
# please download below files from repo_base, save file to data folder
# repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'

img_name = 'elephant-299.jpg'
model_name = 'classify_image_graph_def-with_shapes.pb'
map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'
label_map = 'imagenet_synset_to_human_label_map.txt'

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
target = 'llvm'
target_host = 'llvm'

#layout = "NCHW"
#ctx = tvm.gpu(0)
#target = 'llvm -target=aarch64-linux-gnu -mcpu=cortex-a57'
#target_host = 'llvm -target=aarch64-linux-gnu -mcpu=cortex-a57'
#target = tvm.target.arm_cpu('rk3399')
#target_host = 'arm_cpu'

#target = 'llvm'
#target_host = 'llvm'

layout = None
ctx = tvm.cpu(0)

######################################################################
# Download required files
# -----------------------
# Download files listed above.
#from tvm.contrib.download import download_testdata

img_path = 'data/' + img_name
model_path = 'data/' + model_name
map_proto_path = 'data/' + map_proto
label_path = 'data/' + label_map

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

with tf_compat_v1.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

from PIL import Image
image = Image.open(img_path).resize((299, 299))

x = np.array(image)

shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target,
                                     target_host,
                                     params=params)

print('lib:',lib)

temp = util.tempdir()
path_lib = temp.relpath("deploy_lib.tar")


print('temp path:',temp.relpath(''))

print('path_lib:',path_lib)

lib.export_library(path_lib)

#print(dir(graph),graph)
with open(temp.relpath("deploy_graph.json"), "w") as fo:
    fo.write(graph)
with open(temp.relpath("deploy_param.params"), "wb") as fo:
    fo.write(tvm.relay.save_param_dict(params))
print(temp.listdir())

os.system('rm -rf deploy; cp -a ' + temp.relpath('') + ' deploy')


print('------------------------------------')
print('build network done ...')
print('------------------------------------') 

                                    


