from src.models.recognition.face_recognition.arcfaceresnet100.model import ArcFaceResnet100Model
from src.models.recognition.face_recognition.arcfaceresnet100.postprocess import ArcFaceResnet100Postprocess
from src.models.recognition.face_recognition.arcfaceresnet100.preprocess import ArcFaceResnet100Preprocess


class ArcFaceResnet100:
    def __init__(self, input_queue, model_path, targets, face_recognition_threshold, output_queue=None,
                 kill_flag=None):
        self.input_queue = input_queue
        self.preprocess = ArcFaceResnet100Preprocess(input_queue, kill_flag=kill_flag)
        self.model = ArcFaceResnet100Model(self.preprocess.output_queue, model_path,
                                           kill_flag=kill_flag)
        self.postprocess = ArcFaceResnet100Postprocess(self.model.output_queue, targets,
                                                       face_recognition_threshold, kill_flag=kill_flag,
                                                       output_queue=output_queue)
        self.output_queue = self.postprocess.output_queue

    def start(self):
        self.preprocess.start()
        self.model.start()
        self.postprocess.start()
        return self

    def join(self):
        self.preprocess.join()
        self.model.join()
        self.postprocess.join()

    def __iter__(self):
        return self

    def __next__(self):
        return self.output_queue.get()

    def infer_synchronous(self, x):
        return self.postprocess.infer(self.model.infer(self.preprocess.infer(x)[0])[0])[0]
