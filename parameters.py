import torch
import os
import numpy as np
import torch.nn as nn
import torch.utils.data as utils


class Parameters:
    def __init__(self):
        self.load_weights = 1
        self.model_path = os.path.join('best.pt')


def classify_and_detect(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes
    param = Parameters()

    if param.load_weights:
        print("loading...")

        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=param.model_path)
        print("testing...")

    with torch.no_grad():
        for i, image in enumerate(images):

            image = image.squeeze().reshape((64, 64, 3)).astype(np.uint8)
            detection = torch.round(model(image).xyxy[0])
            if len(detection) == 0:
                cls1 = 0
                cls2 = cls1
                xmin1, ymin1, xmax1, ymax1, conf = 0, 0, 0, 0, 0
                xmin2, ymin2, xmax2, ymax2, conf2 = xmin1, ymin1, xmax1, ymax1, conf

            elif len(detection) == 1:
                cls1 = detection[0, 5].cpu().numpy()
                cls2 = cls1
                xmin1, ymin1, xmax1, ymax1, conf = detection[0][0:5].cpu(
                ).numpy()
                xmin2, ymin2, xmax2, ymax2, conf2 = xmin1, ymin1, xmax1, ymax1, conf

            else:
                cls1 = detection[0, 5].cpu().numpy()
                cls2 = detection[1, 5].cpu().numpy()
                xmin1, ymin1, xmax1, ymax1, conf = detection[0][0:5].cpu(
                ).numpy()
                xmin2, ymin2, xmax2, ymax2, conf2 = detection[1][0:5].cpu(
                ).numpy()

            if cls1 > cls2:
                pred_class[i, 1] = cls1
                pred_class[i, 0] = cls2
                pred_bboxes[i, 1, :] = np.array([ymin1, xmin1, ymax1, xmax1])
                pred_bboxes[i, 0, :] = np.array([ymin2, xmin2, ymax2, xmax2])
            else:
                pred_class[i, 0] = cls1
                pred_class[i, 1] = cls2
                pred_bboxes[i, 0, :] = np.array([ymin1, xmin1, ymax1, xmax1])
                pred_bboxes[i, 1, :] = np.array([ymin2, xmin2, ymax2, xmax2])

    return pred_class, pred_bboxes

