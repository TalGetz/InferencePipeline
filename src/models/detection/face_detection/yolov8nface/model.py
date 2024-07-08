from src.models.base_model import BaseModel
from src.models.detection.face_detection.yolov8nface.item import Item
from src.processes.t_process import TProcess


class YOLOv8nFaceModel(TProcess):
    def __init__(self, input_queue, output_queue_capacity, model_path):
        super().__init__(input_queue, output_queue_capacity)
        self.model = BaseModel(model_path, output_shapes=[
            (1, 80, 80, 80),
            (1, 80, 40, 40),
            (1, 80, 20, 20),
        ])

    def infer(self, item: Item):
        outputs = self.model.infer([item.blob])
        outputs = [outputs[1], outputs[2], outputs[0]]
        item.outputs = outputs
        return item
