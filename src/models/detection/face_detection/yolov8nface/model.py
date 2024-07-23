from src.models.base_model import BaseModel
from src.models.detection.face_detection.yolov8nface.item import YOLOv8nFaceItem
from src.processes.t_process import TProcess
from src.utils.stopwatch import StopWatch


class YOLOv8nFaceModel(TProcess):
    def __init__(self, input_queue, model_path, kill_flag=None):
        super().__init__(input_queue, kill_flag=kill_flag)
        self.model_path = model_path
        self.model: BaseModel = None

    def init_in_process(self):
        self.model = BaseModel(self.model_path, output_shapes=[
            (1, 80, 80, 80),
            (1, 80, 40, 40),
            (1, 80, 20, 20),
        ])

    def overridable_infer(self, item: YOLOv8nFaceItem):
        with StopWatch() as sw:
            outputs = self.model.infer([item.blob])
            outputs = [outputs[1], outputs[2], outputs[0]]
            item.outputs = outputs
        item.model_time = sw.get_time_in_seconds()
        return [item]
