import cv2
import numpy as np
from skimage import transform as trans

src = np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                [33.5493, 92.3655], [62.7299, 92.2041]],
               dtype=np.float32)
src[:, 0] += 8.0


def align_face_np(img, landmarks):
    landmarks = landmarks.reshape(5, -1)[:, [0, 1]]
    landmark = landmarks.reshape((5, 2))

    assert landmark.shape[0] == 5

    tform = trans.SimilarityTransform()
    tform.estimate(landmark, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return img
