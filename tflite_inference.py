import colorsys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from utils.utils import letterbox_image
from utils.setup_tool import get_classes,get_anchors
import cv2

img = "test_data/london.jpg"
image = Image.open(img)
model_image_size = (416 , 416)

image_shape = ( image.size[1], image.size[0] , 3)

model_image_size[0]%32 == 0, 'Multiples of 32 required'
model_image_size[1]%32 == 0, 'Multiples of 32 required'
boxed_image = letterbox_image(image,tuple(reversed(model_image_size)))

image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)

#print(image.size)
print(image_shape)
print(image_data.shape)



# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_data/small_mobilenet_yolo.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#print(output_details)

# Test model on random input data.
input_shape = input_details[0]['shape']
#print(input_details[0]['shape'])
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

start = timer()

input_data = image_data
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data1 = interpreter.get_tensor(output_details[0]['index'])
output_data2 = interpreter.get_tensor(output_details[1]['index'])
output_data3 = interpreter.get_tensor(output_details[2]['index'])
print(output_data1.shape)
print(output_data2.shape)
print(output_data3.shape)

end = timer()
print(end - start)

start = timer()

input_data = image_data
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data1 = interpreter.get_tensor(output_details[0]['index'])
output_data2 = interpreter.get_tensor(output_details[1]['index'])
output_data3 = interpreter.get_tensor(output_details[2]['index'])
print(output_data1.shape)
print(output_data2.shape)
print(output_data3.shape)

end = timer()
print(end - start)





#print(output_data2.shape)
#print(output_data3.shape)
#print(output_details)


#outs = []

#output_data1 = np.reshape(output_data1 , (1,13, 13, 3, 25)) 
#output_data2 = np.reshape(output_data2 , (1,26, 26, 3, 25)) 
#output_data3 = np.reshape(output_data3 , (1,52, 52, 3, 25)) 
#outs.append(output_data1)
#outs.append(output_data2)
#outs.append(output_data3)