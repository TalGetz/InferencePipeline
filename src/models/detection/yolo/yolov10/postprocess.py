import numpy as np

from src.processes.t_process import TProcess
from src.utils.stopwatch import StopWatch


class YOLOv10Postprocess(TProcess):
    def __init__(self, input_queue, conf_threshold, kill_flag=None):
        super().__init__(input_queue, kill_flag=kill_flag)
        self.conf_threshold = conf_threshold
        self.input_height = 640
        self.input_width = 640

    def overridable_infer(self, item):
        with StopWatch() as sw:
            bboxes, confidences, classIds = self.postprocess(item.outputs, item.scale_h, item.scale_w,
                                                                        item.pad_h, item.pad_w)
            item.detection_bboxes = bboxes
            item.detection_confidences = confidences
            item.detection_class_id = classIds
        item.detection_postprocess_time = sw.get_time_in_seconds()
        return [item]

    def postprocess(self, predictions, scale_h, scale_w, pad_h, pad_w):
        batch_bboxes, batch_scores, batch_classes = [], [], []
        for i, pred in enumerate(predictions):
            bboxes = pred[..., : 4]
            scores = pred[..., 4]
            classes = pred[..., 5]

            bboxes -= np.array([[pad_w, pad_h, pad_w, pad_h]])
            bboxes /= np.array([[scale_w, scale_h, scale_w, scale_h]])

            x1 = bboxes[..., 0]
            y1 = bboxes[..., 1]
            x2 = bboxes[..., 2]
            y2 = bboxes[..., 3]

            x1 = np.clip(x1, 0, self.input_width)
            y1 = np.clip(y1, 0, self.input_height)
            x2 = np.clip(x2, 0, self.input_width)
            y2 = np.clip(y2, 0, self.input_height)
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)

            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
            batch_classes.append(classes)

        batch_bboxes = np.concatenate(batch_bboxes, axis=0)
        batch_scores = np.concatenate(batch_scores, axis=0)
        batch_classes = np.concatenate(batch_classes, axis=0)

        mask = batch_scores > self.conf_threshold
        batch_scores = batch_scores[mask]
        batch_classes = batch_classes[mask]
        batch_bboxes = batch_bboxes[mask]

        return batch_bboxes, batch_scores, batch_classes
