import cv2
import numpy as np
import math

from src.models.base_model import BaseModel
from src.utils import softmax


class YOLOv8nFace(BaseModel):
    def __init__(self, runner, conf_threshold, iou_threshold):
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
        super().__init__(runner,
                         output_shapes=[
                             (1, 80, 80, 80),
                             (1, 80, 40, 40),
                             (1, 80, 20, 20),
                         ]
                         )

    def detect(self, frame):
        blob, pad_h, pad_w, scale_h, scale_w = self.preprocess(frame)
        outputs = self.infer([blob])
        outputs = [outputs[1], outputs[2], outputs[0]]
        det_bboxes, det_conf, det_classid, landmarks = self.postprocess(outputs, scale_h, scale_w, pad_h, pad_w)
        return det_bboxes, det_conf, det_classid, landmarks

    def preprocess(self, frame):
        input_img, newh, neww, pad_h, pad_w = self._resize_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = frame.shape[0] / newh, frame.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0
        blob = input_img.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)
        return blob, pad_h, pad_w, scale_h, scale_w

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

    def _resize_image(self, srcimg, keep_ratio=True):
        pad_top, pad_left, new_h, new_w = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                new_h, new_w = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (new_w, new_h), interpolation=cv2.INTER_AREA)
                pad_left = int((self.input_width - new_w) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, pad_left, self.input_width - new_w - pad_left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
            else:
                new_h, new_w = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (new_w, new_h), interpolation=cv2.INTER_AREA)
                pad_top = int((self.input_height - new_h) * 0.5)
                img = cv2.copyMakeBorder(img, pad_top, self.input_height - new_h - pad_top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, new_h, new_w, pad_top, pad_left
