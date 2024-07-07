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

    if landmark.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36] + landmark[39]) / 2
        landmark5[1] = (landmark[42] + landmark[45]) / 2
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return img
