import torch
import torch.nn as nn
import torchvision
import os
import PIL
import cv2 as cv
import matplotlib.pyplot as plt

labels = ["background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

PATH_SSD_MODEL = os.path.join(
    os.environ["DL_MODELS"], "ssdlite320_mobilenet_v3.ckpt")

PATH_FASTERRCNN_MODEL = os.path.join(
    os.environ["DL_MODELS"], "fasterrcnn_mobilenet_v3.ckpt")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_boxes(I, objs, thresh=0.8):
    I_ = I.copy()
    for box_, label_, score_ in objs:
        if score_ < thresh:
            continue
        x1, y1, x2, y2 = box_
        I_ = cv.rectangle(I_, (int(x1), int(y1)),
                          (int(x2), int(y2)), (255, 0, 0), 2)
        I_ = cv.putText(I_, labels[label_], (int(x1), int(
            y1)), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))

    return I_


class ObjectDetection:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn().to(dev)
        self.model.load_state_dict(torch.load(
            PATH_FASTERRCNN_MODEL, map_location=dev))

        self.model.eval()

    def get_objects(self, I):
        """
        inputs:
            I: numpy ndarray (H,W,C)
        """
        x = torch.from_numpy(I.transpose(2, 0, 1)[
                             None, :, :, :] / 255.0).float().to(dev)
        with torch.no_grad():
            pred = self.model(x)[0]
        return list(zip(pred['boxes'].detach().cpu().numpy(), pred['labels'].detach().cpu().numpy(), pred['scores'].detach().cpu().numpy()))


if __name__ == "__main__":
    od = ObjectDetection()
    I = cv.imread("./samples/desktop.jpeg")[:, :, (2, 1, 0)]
    objs = od.get_objects(I)
    I_ = draw_boxes(I, objs, thresh=0.8)
    plt.imshow(I_)
    plt.show()
