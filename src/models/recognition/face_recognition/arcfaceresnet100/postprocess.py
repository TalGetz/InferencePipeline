import numpy as np

from src.models.recognition.face_recognition.arcfaceresnet100.item import ArcFaceResnet100Item
from src.processes.t_process import TProcess


class ArcFaceResnet100Postprocess(TProcess):
    def __init__(self, input_queue, output_queue_capacity, targets, face_recognition_threshold):
        super().__init__(input_queue, output_queue_capacity)
        self.targets = targets
        self.target_embeddings = [target.face_embedding_batch for target in targets]
        self.face_recognition_threshold = face_recognition_threshold

    def overridable_infer(self, item: ArcFaceResnet100Item):
        similarities = (self.target_embeddings / np.linalg.norm(self.target_embeddings)) @ \
                       (item.face_embedding_batch / np.linalg.norm(item.face_embedding_batch)).T
        return ["hello"]
