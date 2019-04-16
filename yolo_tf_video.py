import tensorflow as tf
from yolo_tf import detect_video
from PIL import Image
import numpy as np
import cv2

from tensorflow.python.platform import gfile

# parameter ==========================
wkdir = 'keras_to_tensorflow'
pb_filename = 'small_mobilenet_trained_model.pb'

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

    detect_video( sess, "test_data/akiha.mp4")