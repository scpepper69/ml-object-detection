import numpy as np
import os
import cv2
from graphpipe import remote

import pyws as m
import win32gui
import time

from PIL import ImageGrab

color=(255, 255, 0)
#cap = cv2.VideoCapture(0)
height,width = 800,800

handle = win32gui.FindWindow(None, 'LonelyScreen AirPlay Receiver')
rect = win32gui.GetWindowRect(handle)

time.sleep(1)

def draw_box(img, box, color, score):
    x, y, w, h = box
    label = 'face '+str(int(score*100))+'%'  #self.name
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1.5, 1)
    cv2.rectangle(img, (x, y), (x + label_size[0], y + label_size[1] + base_line), color, cv2.FILLED)
    cv2.putText(img, label, (x, y + label_size[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0))

def main():
    while(True):

        img = ImageGrab.grab(rect)
        img = np.asarray(img)
        image_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#        cv2.imwrite('path/' + datetime.now().strftime("%Y%m%d%H%M%S") + '.jpg', img)

#        cv2.imshow("camera window", frame)
#        ret, frame = cap.read()

#        image_np = cv2.resize(frame,(width,height))
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        boxes,scores,num_detections,classes = remote.execute_multi(
            'http://127.0.0.1:9022',
            [image_np_expanded],
            ['image_tensor'],
            ['detection_boxes', 'detection_scores', 'num_detections', 'detection_classes'])

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
                draw_box(image_np, (x, y, w, h), color, scores[0][i])

        cv2.putText(image_np, "There are " + str(num_persons) + " persons.", (0, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.imshow("camera window", image_np)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
