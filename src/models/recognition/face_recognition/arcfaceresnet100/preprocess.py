import cv2
import numpy as np
from skimage import transform as trans

from src.processes.t_process import TProcess


class ArcFaceResnet100Preprocess(TProcess):
    ArcFacePoints = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                              [33.5493, 92.3655], [62.7299, 92.2041]],
                             dtype=np.float32)
    ArcFacePoints[:, 0] += 8.0

    def __init__(self, input_queue, kill_flag=None):
        self.input_height = 640
        self.input_width = 640
        super().__init__(input_queue, kill_flag=kill_flag)

    def overridable_infer(self, item):
        item.aligned_face_batch = np.array(
            [self.align_face_np(item.frame, landmarks).transpose((2, 0, 1)) for landmarks in item.landmarks]
        )
        return [item]

    @staticmethod
    def align_face_np(img, landmarks):
        landmarks = landmarks.reshape(5, -1)[:, [0, 1]]
        landmark = landmarks.reshape((5, 2))

        tform = trans.SimilarityTransform()
        tform.estimate(landmark, ArcFaceResnet100Preprocess.ArcFacePoints)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        return img
