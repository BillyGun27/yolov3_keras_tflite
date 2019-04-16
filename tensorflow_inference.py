# # load & inference the model ==================


import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from utils.utils import letterbox_image
np.random.seed(0)

from tensorflow.python.platform import gfile

# parameter ==========================
wkdir = 'keras_to_tensorflow'
pb_filename = 'model.pb'

img = "test_data/london.jpg"
image = Image.open(img)

#image_shape = ( image.size[1], image.size[0] , 3)
model_image_size = (416 , 416)

model_image_size[0]%32 == 0, 'Multiples of 32 required'
model_image_size[1]%32 == 0, 'Multiples of 32 required'
boxed_image = letterbox_image(image,tuple(reversed(model_image_size)))

image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)

#print(image.size)
#print(image_shape)
print(image_data.shape)

#x = np.random.rand(1,416,416,3)
#y = np.vstack((np.ones((1000,1)),np.zeros((1000,1))))
#print(x.shape)
#print(y.shape)

#setup graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  with gfile.FastGFile(wkdir+'/'+pb_filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def)



#detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
  
    start = timer()

    tensor_output1 = sess.graph.get_tensor_by_name('import/conv2d_5/convolution:0')
    tensor_output2 = sess.graph.get_tensor_by_name('import/conv2d_11/convolution:0')
    tensor_output3 = sess.graph.get_tensor_by_name('import/conv2d_17/convolution:0')
    tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')

    predictions1,  predictions2 , predictions3 = sess.run([tensor_output1,tensor_output2,tensor_output3], {tensor_input: image_data})
    print('\n===== output predicted results =====\n')
    print(predictions1.shape)
    print(predictions2.shape)
    print(predictions3.shape)

    end = timer()
    print(end - start)
  
    start = timer()

    tensor_output1 = sess.graph.get_tensor_by_name('import/conv2d_5/convolution:0')
    tensor_output2 = sess.graph.get_tensor_by_name('import/conv2d_11/convolution:0')
    tensor_output3 = sess.graph.get_tensor_by_name('import/conv2d_17/convolution:0')
    tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')

    predictions1,  predictions2 , predictions3 = sess.run([tensor_output1,tensor_output2,tensor_output3], {tensor_input: image_data})
    print('\n===== output predicted results =====\n')
    print(predictions1.shape)
    print(predictions2.shape)
    print(predictions3.shape)

    end = timer()
    print(end - start)