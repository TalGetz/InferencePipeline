import time

from src.models.base_model import BaseModel
from src.models.detection.face_detection.yolov8nface.item import YOLOv8nFaceItem
from src.processes.t_process import TProcess


class YOLOv8nFaceModel(TProcess):
    def __init__(self, input_queue, output_queue_capacity, model_path):
        super().__init__(input_queue, output_queue_capacity)
        self.model_path = model_path
        self.model = None

    def init_in_process(self):
        self.model = BaseModel(self.model_path, output_shapes=[
            (1, 80, 80, 80),
            (1, 80, 40, 40),
            (1, 80, 20, 20),
        ])

    def infer(self, item: YOLOv8nFaceItem):
        outputs = self.model.infer_synchronous([item.blob])
        outputs = [outputs[1], outputs[2], outputs[0]]
        item.outputs = outputs
        return item
