import numpy as np

from src.models.recognition.face_recognition.arcfaceresnet100.item import ArcFaceResnet100Item
from src.processes.t_process import TProcess


class ArcFaceResnet100Postprocess(TProcess):
    def __init__(self, input_queue, output_queue_capacity, targets, face_recognition_threshold):
        super().__init__(input_queue, output_queue_capacity)
        self.targets = targets
        if len(self.targets) > 0 and self.targets[0].face_embedding_batch is not None:
            self.target_embeddings = np.stack([t.face_embedding_batch[0] for t in self.targets]) if len(
                self.targets) > 0 else np.ndarray((0, 512))
        self.face_recognition_threshold = face_recognition_threshold

    def overridable_infer(self, item: ArcFaceResnet100Item):
        normalized_target_embeddings = self.target_embeddings / np.expand_dims(
            np.linalg.norm(self.target_embeddings, axis=1), axis=1)
        normalized_item_embeddings = item.face_embedding_batch / np.expand_dims(
            np.linalg.norm(item.face_embedding_batch, axis=1), axis=1)
        similarities = (normalized_item_embeddings @ normalized_target_embeddings.T)
        if similarities.shape[-1] == 0:
            return []
        item.similarities = similarities
        item.highest_target_similarity_index = np.argmax(similarities, axis=1)
        item.is_above_similarity_threshold = item.similarities[np.arange(item.similarities.shape[0]),
        item.highest_target_similarity_index] > self.face_recognition_threshold
        item.matched_names = []
        for index in range(len(item.is_above_similarity_threshold)):
            if item.is_above_similarity_threshold[index]:
                item.matched_names.append(self.targets[item.highest_target_similarity_index[index]].name)
            else:
                item.matched_names.append("")
        return [item]
