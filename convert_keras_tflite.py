from tensorflow.contrib import lite

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

converter = lite.TFLiteConverter.from_keras_model_file('model_data/small_mobilenet_trained_model.h5') # Your model's name
model = converter.convert()
file = open( 'model_data/small_mobilenet_yolo.tflite' , 'wb' ) 
file.write( model )