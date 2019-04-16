from tensorflow.contrib import lite

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#model_name = "416bnfuse_small_mobilenets2_trained_model"
model_name = "416bnfuse_tiny_yolo"
#model_name = "224small_mobilenets2_trained_model"

converter = lite.TFLiteConverter.from_keras_model_file('model_data/'+model_name+'.h5' ) # Your model's name

model = converter.convert()
file = open( 'model_data/'+model_name+'.tflite' , 'wb' ) 
file.write( model )

#converter.post_training_quantize=True
#tflite_quantized_model=converter.convert()
#open("model_data/quantized_small_mobilenet_yolo.tflite", "wb").write(tflite_quantized_model)