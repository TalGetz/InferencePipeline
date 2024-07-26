import cv2
import numpy as np
import torch

from src.models.detection.item import DetectionItem
from src.processes.compute_thread import ComputeThread
from src.utils.stopwatch import StopWatch


class YOLOv10Preprocess(ComputeThread):
    def __init__(self, input_queue, kill_flag=None, output_queue=None):
        self.input_height = 640
        self.input_width = 640
        super().__init__(input_queue, kill_flag=kill_flag, output_queue=output_queue)

    def overridable_infer(self, frame: np.ndarray):
        with StopWatch() as sw:
            blob, pad_h, pad_w, scale_h, scale_w = self.preprocess(frame)
            item = DetectionItem(frame)
            item.blob = blob
            item.pad_h = pad_h
            item.pad_w = pad_w
            item.scale_h = scale_h
            item.scale_w = scale_w
        item.detection_preprocess_time = sw.get_time_in_seconds()
        return [item]

    def preprocess(self, image):
        """
        Prepares input image before inference.

        Args:
            images (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        new_shape = self.input_height, self.input_width
        image, pad_h, pad_w, scale_h, scale_w = self.resize_and_pad(image, new_shape)
        image = image[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW, (n, 3, h, w)
        image = np.ascontiguousarray(image)  # contiguous
        image = torch.from_numpy(image)
        image = image.float()

        image /= 255  # 0 - 255 to 0.0 - 1.0
        return image, pad_h, pad_w, scale_h, scale_w

    def resize_and_pad(self, image, new_shape):
        shape = image.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top = int(round(dh / 2 - 0.1))
        bottom = dh - top
        left = int(round(dw - 0.1))
        right = dw - left
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return image, top, left, new_unpad[1] / shape[0], new_unpad[0] / shape[1]
