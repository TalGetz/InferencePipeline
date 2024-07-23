import numpy as np

from src.models.base_model import BaseModel
from src.processes.t_process import TProcess
from src.utils.stopwatch import StopWatch


class ArcFaceResnet100Model(TProcess):
    def __init__(self, input_queue, model_path, kill_flag=None):
        super().__init__(input_queue, kill_flag=kill_flag)
        self.model_path = model_path
        self.model: BaseModel = None

    def init_in_process(self):
        self.model = BaseModel(self.model_path, output_shapes=[
            (512,),
        ])

    def overridable_infer(self, item):
        with StopWatch() as sw:
            embeddings = []
            for aligned_face in item.aligned_face_batch:
                [embedding] = self.model.infer([aligned_face])
                embeddings.append(embedding)

            if len(embeddings) > 0:
                item.face_embedding_batch = np.stack(embeddings)
            else:
                item.face_embedding_batch = np.ndarray((0, 512))
        item.face_recognition_model_time = sw.get_time_in_seconds()
        return [item]
