import cv2
import numpy as np

from src.models.detection.face_detection.yolov8nface.item import YOLOv8nFaceItem
from src.processes.t_process import TProcess
from src.utils.stopwatch import StopWatch


class YOLOv8nFacePreprocess(TProcess):
    def __init__(self, input_queue, kill_flag=None):
        self.input_height = 640
        self.input_width = 640
        super().__init__(input_queue, kill_flag=kill_flag)

    def overridable_infer(self, frame: np.ndarray):
        with StopWatch() as sw:
            blob, pad_h, pad_w, scale_h, scale_w = self.preprocess(frame)
            item = YOLOv8nFaceItem(frame)
            item.blob = blob
            item.pad_h = pad_h
            item.pad_w = pad_w
            item.scale_h = scale_h
            item.scale_w = scale_w
        item.preprocess_time = sw.get_time_in_seconds()
        return [item]

    def preprocess(self, frame):
        input_img, newh, neww, pad_h, pad_w = self._resize_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = frame.shape[0] / newh, frame.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0
        blob = input_img.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)
        return blob, pad_h, pad_w, scale_h, scale_w

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
