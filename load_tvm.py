#----------------------------------------
# runtime part
# please install tvm runtime before test 
#----------------------------------------

import tvm
from tvm import te
from tvm import relay
import time
from tvm.contrib import util,graph_runtime

import numpy as np
import os.path
import tvm
from tvm import te
from PIL import Image
import tvm.relay.testing.tf as tf_testing

layout = None
ctx = tvm.cpu(0)


img_name = 'elephant-299.jpg'
label_map = 'imagenet_synset_to_human_label_map.txt'
map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'


img_path = 'data/' + img_name
#model_path = 'data/' + model_name
map_proto_path = 'data/' + map_proto
label_path = 'data/' + label_map

image = Image.open(img_path).resize((299, 299))
x = np.array(image)

shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}



loaded_json = open("deploy/deploy_graph.json").read()
loaded_lib = tvm.runtime.load_module("deploy/deploy_lib.tar")
params_bytes = bytearray(open("deploy/deploy_param.params","rb").read())
loaded_params = tvm.relay.load_param_dict(params_bytes)
m = graph_runtime.create(loaded_json, loaded_lib, ctx)

# set param
m.set_input(**loaded_params)

print('1.set inputs')
# set input
m.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype('uint8')))

#m.set_input(**params)


t1 = time.time()
print('2.execute')
# execute
m.run()


print('3.get outputs')
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))


print('4.proces the output')
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
node_lookup = tf_testing.NodeLookup(label_lookup_path=map_proto_path,
                                    uid_lookup_path=label_path)
print('5.print output')
# Print top 5 predictions from TVM output.
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))

t2 = time.time()
print('tvm inference time:',t2-t1)


