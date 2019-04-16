import colorsys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from utils.utils import letterbox_image
from utils.setup_tool import get_classes,get_anchors
import cv2

#model_path="model_data/small_mobilenet_yolo.tflite"
#model_path="model_data/tiny_yolo.tflite"
#model_path="model_data/mobilenet_trained_model.tflite"
#model_path="model_data/mobilenetv2_trained_model.tflite"

score_thres = 0.3 #obj, score
iou_thres = 0.45 #nms, iou
model_image_size = (416 , 416)
#model_image_size = (224, 224)

classes_path = 'class/voc_classes.txt'
anchors_path = 'anchors/yolo_anchors.txt'

#classes_path = 'class/coco_classes.txt'
#anchors_path = 'anchors/tiny_yolo_anchors.txt'


class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)

num_classes = len(class_names)
num_anchors = len(anchors)

num_layers = num_anchors//3

if num_layers==3 :
    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] 
elif num_layers==2 :
    masks =  [[3,4,5], [0,1,2]]
else :
    masks = [[0,1,2]]

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

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path=model_path)
#interpreter.allocate_tensors()

# Get input and output tensors.
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()

def sigmoid(x):
    """sigmoid.

    # Arguments
        x: Tensor.

    # Returns
        numpy ndarray.
    """
    return 1 / (1 + np.exp(-x))

def process_feats( out, anchors, mask):
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
    box_xy = sigmoid(out[..., :2])
    box_wh = np.exp(out[..., 2:4])
    box_wh = box_wh * anchors_tensor

    box_confidence = sigmoid(out[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = sigmoid(out[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= model_image_size #(416, 416)
    box_xy -= (box_wh / 2.)
    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    return boxes, box_confidence, box_class_probs

def filter_boxes( boxes, box_confidences, box_class_probs):
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
    pos = np.where(box_class_scores >= score_thres)

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
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep

def yolo_out( outs , shape):
    """Process output of yolo base net.

    # Argument:
        outs: output of yolo base net.
        shape: shape of original image.

    # Returns:
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
    """

    boxes, classes, scores = [], [], []

    for out, mask in zip(outs, masks):
        b, c, s = process_feats(out, anchors, mask)
        b, c, s = filter_boxes(b, c, s)
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

def tf_out(sess, image_data):

    fmap = model_image_size[0]//32
    mapsize = [1,2,4]

    outs = []

    tensor_output1 = sess.graph.get_tensor_by_name('import/conv2d_5/convolution:0')
    tensor_output2 = sess.graph.get_tensor_by_name('import/conv2d_11/convolution:0')
    tensor_output3 = sess.graph.get_tensor_by_name('import/conv2d_17/convolution:0')
    
    #tensor_output1 = sess.graph.get_tensor_by_name('import/conv2d_7/convolution:0') #v2
    #tensor_output2 = sess.graph.get_tensor_by_name('import/conv2d_15/convolution:0') #v2
    #tensor_output3 = sess.graph.get_tensor_by_name('import/conv2d_23/convolution:0') #v2

    #tensor_output1 = sess.graph.get_tensor_by_name('import/conv2d_10/convolution:0') #tiny
    #tensor_output2 = sess.graph.get_tensor_by_name('import/conv2d_13/convolution:0') #tiny

    tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')

    predictions1,  predictions2 , predictions3 = sess.run([tensor_output1,tensor_output2,tensor_output3], {tensor_input: image_data})
    #predictions1,  predictions2 = sess.run([tensor_output1,tensor_output2], {tensor_input: image_data})#tiny
    #print('\n===== output predicted results =====\n')
    #print(predictions1.shape)
    #print(predictions2.shape)
    #print(predictions3.shape)

    predictions1= np.reshape(predictions1 , (1, fmap*mapsize[0], fmap*mapsize[0] , 3 , (num_classes + 5) ) ) 
    predictions2= np.reshape(predictions2 , (1, fmap*mapsize[1], fmap*mapsize[1] , 3 , (num_classes + 5) ) ) 
    predictions3= np.reshape(predictions3 , (1, fmap*mapsize[2], fmap*mapsize[2] , 3 , (num_classes + 5) ) ) 
    
    outs.append(predictions1)
    outs.append(predictions2)
    outs.append(predictions3)

    return outs

def detect_image(sess, image):
    start = timer()

    image_shape = ( image.size[1], image.size[0] , 3)

    model_image_size[0]%32 == 0, 'Multiples of 32 required'
    model_image_size[1]%32 == 0, 'Multiples of 32 required'
    boxed_image = letterbox_image(image,tuple(reversed(model_image_size)))

    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    #print(image.size)
    #print(image_shape)
    #print(image_data.shape)

    outs = tf_out(sess,image_data)
    
    out_boxes, out_classes, out_scores = yolo_out( outs , image_shape )

    print(model_image_size)
    if not out_boxes is  None :
        print('Found {} boxes for {}'.format(len( out_boxes ), 'img'))
    
        #print(out_boxes)

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

    else : 
         print('No Boxes')

    end = timer()
    print(end - start)
    return image

def detect_video( sess , video_path, output_path="" ):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = detect_image(sess , image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        height, width, channels = result.shape
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('result', width,height) 
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break