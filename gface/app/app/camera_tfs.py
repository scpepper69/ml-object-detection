import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #disable gpu
import sys
import tensorflow as tf
import cv2
import time
from PIL import Image

from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2,predict_pb2

color=(255, 255, 0)
cap = cv2.VideoCapture(0)
height,width = 600,800

#TF Serving host
SERVING_HOST='172.30.233.208'
SERVING_PORT=8500

# create grpc stub
channel = implementations.insecure_channel(SERVING_HOST, SERVING_PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# Create prediction request object
request = predict_pb2.PredictRequest()

# Specify model name (must be the same as when the TensorFlow serving serving was started)
request.model_spec.name = 'gface_tfs'

# Initalize prediction 
request.model_spec.signature_name = "serving_default"

class_array = ['None','Gundam','Zaku']
def ret_class(n):
    return class_array[n]

def draw_box(img, box, color, score, target_class):
    x, y, w, h = box
    label = str(ret_class(int(target_class)))+' face '+str(int(score*100))+'%'  #self.name
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.5, 1)
    cv2.rectangle(img, (x, y), (x + label_size[0], y + label_size[1] + base_line), color, cv2.FILLED)
    cv2.putText(img, label, (x, y + label_size[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0))

def main():
    while(True):
        ret, frame = cap.read()

        image_np = cv2.resize(frame,(width,height))
        image_np_expanded = np.expand_dims(image_np, axis=0)

        starttime = time.time()

        # Input Definition
        request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(image_np_expanded))

        # Actual detection.
        result = stub.Predict(request, 10.0)  # 10 secs timeout

        boxes = result.outputs['detection_boxes'].float_val
        classes = result.outputs['detection_classes'].float_val
        scores = result.outputs['detection_scores'].float_val

#        image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
#            image_np,
#            np.reshape(boxes,[100,4]),
#            np.squeeze(classes).astype(np.int32),
#            np.squeeze(scores),
#            category_index,
#            use_normalized_coordinates=True,
#            line_thickness=8)

        # Visualize Objects
        num_persons=0
        for i in range(np.reshape(boxes,[100,4]).shape[0]):
            if np.squeeze(scores)[i] >= 0.5:
                num_persons+=1

                im_height, im_width, _ = image_np.shape
                ymin, xmin, ymax, xmax = tuple(np.reshape(boxes,[100,4])[i].tolist())
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                x, y, w, h = int(left), int(top), int(right - left), int(bottom - top)
                draw_box(image_np, (x, y, w, h), color, np.squeeze(scores)[i], np.squeeze(classes).astype(np.int32)[i])

        endtime = time.time()
        interval = endtime - starttime
        print(str(interval) + " sec")

        cv2.imshow("camera window", image_np) 

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
