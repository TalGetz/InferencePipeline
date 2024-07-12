import numpy as np

from src.models.base_model import BaseModel
from src.processes.t_process import TProcess


class ArcFaceResnet100Model(TProcess):
    def __init__(self, input_queue, output_queue_capacity, model_path, kill_flag=None):
        super().__init__(input_queue, output_queue_capacity, kill_flag=kill_flag)
        self.model_path = model_path
        self.model: BaseModel = None

    def init_in_process(self):
        self.model = BaseModel(self.model_path, output_shapes=[
            (512,),
        ])

    def overridable_infer(self, item):
        embeddings = []
        for aligned_face in item.aligned_face_batch:
            [embedding] = self.model.infer([aligned_face])
            embeddings.append(embedding)

        if len(embeddings) > 0:
            item.face_embedding_batch = np.stack(embeddings)
        else:
            item.face_embedding_batch = np.ndarray((0, 512))
        return [item]
