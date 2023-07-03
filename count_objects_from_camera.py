import cv2
from object_counting_api import ObjectCountingAPI

options = {"model": "cfg/tiny-yolo-voc-2.cfg", "load": -1, "threshold": 0.65, "gpu": 1.0}

#
# cap = cv2.VideoCapture(VIDEO_PATH)
cap = cv2.VideoCapture(0) #real-time video

counter = ObjectCountingAPI(options)

counter.count_objects_on_video(cap, show=True)
