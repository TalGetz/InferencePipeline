import cv2
import numpy as np
from skimage import transform as trans

from src.models.detection.face_detection.yolov8nface.item import YOLOv8nFaceItem
from src.models.recognition.face_recognition.arcfaceresnet100.item import ArcFaceResnet100Item
from src.processes.t_process import TProcess


class ArcFaceResnet100Preprocess(TProcess):
    ArcFacePoints = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                              [33.5493, 92.3655], [62.7299, 92.2041]],
                             dtype=np.float32)
    ArcFacePoints[:, 0] += 8.0

    def __init__(self, input_queue, output_queue_capacity, kill_flag=None):
        self.input_height = 640
        self.input_width = 640
        super().__init__(input_queue, output_queue_capacity, kill_flag=kill_flag)

    def overridable_infer(self, item):
        if isinstance(item, YOLOv8nFaceItem):
            new_item = ArcFaceResnet100Item(item.frame, item.landmarks, item.det_bboxes)
        elif isinstance(item, ArcFaceResnet100Item):
            new_item = ArcFaceResnet100Item(item.frame, item.landmarks_batch, item.bbox_batch)
        else:
            raise NotImplementedError()
        new_item.aligned_face_batch = np.array(
            [self.align_face_np(new_item.frame, landmarks).transpose(2,0,1) for landmarks in new_item.landmarks_batch]
        )
        return [new_item]

    @staticmethod
    def align_face_np(img, landmarks):
        landmarks = landmarks.reshape(5, -1)[:, [0, 1]]
        landmark = landmarks.reshape((5, 2))

        tform = trans.SimilarityTransform()
        tform.estimate(landmark, ArcFaceResnet100Preprocess.ArcFacePoints)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        return img
