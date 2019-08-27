import numpy as np
import os
import sys
import cv2
import time
from PIL import Image
from graphpipe import remote

color=(255, 255, 0)
cap = cv2.VideoCapture(0)
height,width = 600,800

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

        # Actual detection.
        boxes,scores,num_detections,classes = remote.execute_multi(
            'http://172.25.172.110:9023',
            [image_np_expanded],
            ['image_tensor'],
            ['detection_boxes', 'detection_scores', 'num_detections', 'detection_classes'])

        endtime = time.time()
        interval = endtime - starttime
        print(str(interval) + " sec")

        # Visualize Objects
        num_persons=0
        for i in range(boxes[0].shape[0]):
            if scores[0][i] >= 0.5:
                num_persons+=1

                im_height, im_width, _ = image_np.shape
                ymin, xmin, ymax, xmax = tuple(boxes[0][i].tolist())
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)

                x, y, w, h = int(left), int(top), int(right - left), int(bottom - top)
                draw_box(image_np, (x, y, w, h), color, scores[0][i], classes[0][i])


        cv2.imshow("camera window", image_np) 
#        cv2.imwrite("gface_detection"+str(endtime)+".jpg",image_np)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
