import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from keras.applications.mobilenet import MobileNet
#from model.yolo3 import tiny_yolo_body
#from model.small_mobilenets2 import yolo_body
#from model.medium_darknet import yolo_body
#from model.mobilenet import yolo_body
from model.yolo3 import yolo_body

run_meta = tf.RunMetadata()
with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    #net = MobileNet(alpha=.75, input_tensor=tf.placeholder('float32', shape=(1,32,32,3)) )
    
    #net = MobileNet(input_tensor=tf.placeholder('float32', shape=(1,416,416,3)) ,weights='imagenet')
    image_input = Input(shape=(416,416, 3))
    #net = tiny_yolo_body(image_input, 3 , 20)
    net = yolo_body(image_input, 3 , 20)

    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("floatops _ {:,} totalparams _ {:,}".format(flops.total_float_ops, params.total_parameters))