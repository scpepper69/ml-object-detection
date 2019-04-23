import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from graphpipe import remote

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

cap = cv2.VideoCapture(0)
height = 400
width  = 400

def main():
    while(True):
        ret, frame = cap.read()

        image_np = cv2.resize(frame,(width,height))
        image_np_expanded = np.expand_dims(image_np, axis=0)

        starttime = time.time()

        pred = remote.execute_multi("http://127.0.0.1:9005", [image_np_expanded], ['image_tensor'], ['detection_boxes', 'detection_scores', 'num_detections', 'detection_classes'])

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            pred[-4][0],
            pred[-1][0].astype(np.uint8),
            pred[-3][0],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        endtime = time.time()
        interval = endtime - starttime
        print(str(interval) + "sec")

        cv2.imshow("camera window", image_np) 

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
