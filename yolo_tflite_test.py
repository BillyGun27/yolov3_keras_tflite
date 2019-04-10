import colorsys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from utils.utils import letterbox_image
from utils.setup_tool import get_classes,get_anchors
import cv2

_t1 = 0.3 #obj, score
_t2 = 0.45 #nms, iou

def _sigmoid(x):
    """sigmoid.

    # Arguments
        x: Tensor.

    # Returns
        numpy ndarray.
    """
    return 1 / (1 + np.exp(-x))

def _process_feats( out, anchors, mask):
    """process output features.

    # Arguments
        out: Tensor (N, N, 3, 4 + 1 +80), output feature map of yolo.
        anchors: List, anchors for box.
        mask: List, mask for anchors.

    # Returns
        boxes: ndarray (N, N, 3, 4), x,y,w,h for per box.
        box_confidence: ndarray (N, N, 3, 1), confidence for per box.
        box_class_probs: ndarray (N, N, 3, 80), class probs for per box.
    """
    grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

    anchors = [anchors[i] for i in mask]
    anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

    # Reshape to batch, height, width, num_anchors, box_params.
    out = out[0]
    box_xy = _sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4])
    box_wh = box_wh * anchors_tensor

    box_confidence = _sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _sigmoid(out[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs

def _filter_boxes( boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= _t1)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def _nms_boxes( boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= _t2)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep

def _yolo_out( out, shape):
    """Process output of yolo base net.

    # Argument:
        outs: output of yolo base net.
        shape: shape of original image.

    # Returns:
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
    """
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                [59, 119], [116, 90], [156, 198], [373, 326]]
    print(anchors[0][0])
    boxes, classes, scores = [], [], []

    for out, mask in zip(outs, masks):
        b, c, s = _process_feats(out, anchors, mask)
        b, c, s = _filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # Scale boxes back to original image shape.
    width, height = shape[1], shape[0]
    image_dims = [width, height, width, height]
    boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = _nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


classes_path = 'class/voc_classes.txt'
anchors_path = 'anchors/yolo_anchors.txt'

class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
print(anchors[0][0])

num_classes = len(class_names)
num_anchors = len(anchors)

anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] 

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

start = timer()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_data/small_mobilenet_yolo.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
#print(input_details[0]['shape'])
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = image_data
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data1 = interpreter.get_tensor(output_details[0]['index'])
output_data2 = interpreter.get_tensor(output_details[1]['index'])
output_data3 = interpreter.get_tensor(output_details[2]['index'])

#print(output_data.shape)
#print(output_data2.shape)
#print(output_details)

outs = []

output_data1 = np.reshape(output_data1 , (1,13, 13, 3, 25)) 
output_data2 = np.reshape(output_data2 , (1,26, 26, 3, 25)) 
output_data3 = np.reshape(output_data3 , (1,52, 52, 3, 25)) 
outs.append(output_data1)
outs.append(output_data2)
outs.append(output_data3)


out_boxes, out_classes, out_scores = _yolo_out( outs , image_shape )

print(out_boxes)
#print(out_classes)
#print(out_scores)

print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

  # Generate colors for drawing bounding boxes.
hsv_tuples = [(x / len(class_names), 1., 1.)
                for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
np.random.seed(10101)  # Fixed seed for consistent colors across runs.
np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
np.random.seed(None)  # Reset seed to default.

font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
thickness = (image.size[0] + image.size[1]) // 300

for i, c in reversed(list(enumerate(out_classes))):
    predicted_class = class_names[c]
    box = out_boxes[i]
    score = out_scores[i]

    label = '{} {:.2f}'.format(predicted_class, score)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    x,y,w,h = box
    top = max(0, np.floor(y + 0.5).astype('int32'))
    left = max(0, np.floor(x + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor( y + h + 0.5).astype('int32')) 
    right = min(image.size[0], np.floor( x + w + 0.5).astype('int32'))
    print(label, (left, top), (right, bottom), ( (right-left) ,(bottom-top) ) )

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=colors[c])
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=colors[c])
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw

end = timer()
print(end - start)

result = np.asarray(image)
#r_image.show()
#print(image.size)
#print(result.shape)
height, width, channels = result.shape
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', width,height) 
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


#pred_xy, pred_wh , pred_conf , pred_class = numpy_yolo_head( output_data ,anchors[anchor_mask[0]], model_image_size )

#pred_detect = np.concatenate([pred_xy, pred_wh , pred_conf , pred_class ],axis=-1)

#print(pred_detect.shape)

#box = np.where(pred_detect[...,4] > 0.5 )
#box = np.transpose(box)

#for k in range(len(box)):
#    print( pred_detect[tuple(box[k])] )

#boxes = numpy_yolo_correct_boxes(pred_xy, pred_wh, model_image_size , image_shape )
#boxes = np.reshape(boxes, [-1, 4])
#box_scores = pred_conf * pred_class
#box_scores = np.reshape(box_scores, [-1, num_classes])

#print(boxes)