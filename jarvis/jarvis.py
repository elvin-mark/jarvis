import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from vision.object_detection import ObjectDetection, draw_boxes
from audio.speech_recog import SpeechRecognition


class Jarvis:
    def __init__(self):
        self.od = ObjectDetection()
        self.sr = SpeechRecognition()


if __name__ == "__main__":
    cam = cv.VideoCapture(0)
    od = ObjectDetection()
    while True:
        _, I = cam.read()
        objs = od.get_objects(I)
        I_ = draw_boxes(I, objs, thresh=0.5)
        cv.imshow("video", I_)
        if cv.waitKey(20) == 27:
            break
