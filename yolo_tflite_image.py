from yolo_tflite import detect_image
from PIL import Image
import numpy as np
import cv2

img = "test_data/london.jpg"
image = Image.open(img)

r_image = detect_image(image)
result = np.asarray(image)
height, width, channels = result.shape
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', width,height) 
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()