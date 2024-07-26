from src.models.detection.face_detection.yolov8nface.model import YOLOv8nFaceModel
from src.models.detection.face_detection.yolov8nface.postprocess import YOLOv8nFacePostprocess
from src.models.detection.face_detection.yolov8nface.preprocess import YOLOv8nFacePreprocess


class YOLOv8nFace:
    def __init__(self, input_queue, model_path, conf_threshold, iou_threshold, output_queue=None, kill_flag=None):
        self.input_queue = input_queue
        self.preprocess = YOLOv8nFacePreprocess(input_queue, kill_flag=kill_flag)
        self.model = YOLOv8nFaceModel(self.preprocess.output_queue, model_path, kill_flag=kill_flag)
        self.postprocess = YOLOv8nFacePostprocess(self.model.output_queue, conf_threshold, iou_threshold,
                                                  kill_flag=kill_flag, output_queue=output_queue)
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
