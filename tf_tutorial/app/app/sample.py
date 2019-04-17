from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import requests
import os
import cv2

from graphpipe import remote

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

image = Image.open("dog.jpg")
image_np = load_image_into_numpy_array(image)

data = np.array(Image.open("dog.jpg"))
data = data.reshape([1] + list(data.shape))

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

pred = remote.execute_multi("http://127.0.0.1:9005", [data], ['image_tensor'],
        ['detection_boxes', 'detection_scores', 'num_detections',
        'detection_classes'])
print("Class predictions: ", pred[-1][0].astype(np.uint8))
print("Class predictions: ", pred[-4][0])
print("Class predictions: ", pred[-3][0])

vis_util.visualize_boxes_and_labels_on_image_array(
  image_np,
  pred[-4][0],
  pred[-1][0].astype(np.uint8),
  pred[-3][0],
  category_index,
  use_normalized_coordinates=True,
  line_thickness=8)

cv2.imwrite("/tmp/test.jpg", image_np)
