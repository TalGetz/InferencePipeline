import math

import cv2
import numpy as np

from src.processes.t_process import TProcess
from src.utils.softmax import softmax


class YOLOv8nFacePostprocess(TProcess):
    def __init__(self, input_queue, output_queue_capacity, conf_threshold, iou_threshold):
        super().__init__(input_queue, output_queue_capacity)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16
        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [
            (
                math.ceil(self.input_height / stride),
                math.ceil(self.input_width / stride)
            ) for stride in self.strides]
        self.anchors = self._make_anchors(self.feats_hw)

    def infer(self, item):
        mlvl_bboxes, confidences, classIds, landmarks = self.postprocess(item.outputs, item.scale_h, item.scale_w,
                                                                         item.pad_h, item.pad_w)
        item.det_bboxes = mlvl_bboxes
        item.det_conf = confidences
        item.det_classid = classIds
        item.landmarks = landmarks
        return item

    def postprocess(self, predictions, scale_h, scale_w, pad_h, pad_w):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(predictions):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))

            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self._distance2bbox(self.anchors[stride], bbox_pred,
                                       max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[pad_w, pad_h, pad_w, pad_h]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([pad_w, pad_h, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold)

        if len(indices) > 0:
            indices = indices.flatten()
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def _distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points
