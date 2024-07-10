from dataclasses import dataclass

import numpy as np


@dataclass
class ArcFaceResnet100Item:
    frame: np.ndarray
    landmarks_batch: np.ndarray

    aligned_face_batch: np.ndarray = None

    face_embedding_batch: np.ndarray = None
