from src.models.base_model import BaseModel
from src.models.detection.item import DetectionItem
from src.processes.compute_thread import ComputeThread
from src.utils.stopwatch import StopWatch


class YOLOv10Model(ComputeThread):
    def __init__(self, input_queue, model_path, kill_flag=None, output_queue=None):
        super().__init__(input_queue, kill_flag=kill_flag, output_queue=output_queue)
        self.model_path = model_path
        self.model: BaseModel = None

    def init_in_process(self):
        self.model = BaseModel(self.model_path, output_shapes=[
            (1, 300, 6),
        ])

    def overridable_infer(self, item: DetectionItem):
        with StopWatch() as sw:
            outputs = self.model.infer([item.blob])
            item.outputs = outputs
        item.detection_model_time = sw.get_time_in_seconds()
        return [item]
