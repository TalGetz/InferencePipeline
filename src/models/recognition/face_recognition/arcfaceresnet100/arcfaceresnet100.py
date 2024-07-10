from src.models.recognition.face_recognition.arcfaceresnet100.model import ArcFaceResnet100Model
from src.models.recognition.face_recognition.arcfaceresnet100.postprocess import ArcFaceResnet100Postprocess
from src.models.recognition.face_recognition.arcfaceresnet100.preprocess import ArcFaceResnet100Preprocess


class ArcFaceResnet100:
    def __init__(self, input_queue, queue_capacity, model_path, targets, face_recognition_threshold):
        self.input_queue = input_queue
        self.preprocess = ArcFaceResnet100Preprocess(input_queue, queue_capacity)
        self.model = ArcFaceResnet100Model(self.preprocess.output_queue, queue_capacity, model_path)
        self.postprocess = ArcFaceResnet100Postprocess(self.model.output_queue,
                                                       queue_capacity, targets, face_recognition_threshold)
        self.output_queue = self.postprocess.output_queue

    def start(self):
        self.preprocess.start()
        self.model.start()
        self.postprocess.start()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.output_queue.get()

    def infer_synchronous(self, x, get_only_embedding=False):
        if get_only_embedding:
            return self.model.infer(self.preprocess.infer(x)[0])[0]
        return self.postprocess.infer(self.model.infer(self.preprocess.infer(x)[0])[0])[0]