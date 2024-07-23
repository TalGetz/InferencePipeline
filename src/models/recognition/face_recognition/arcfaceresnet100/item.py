from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ArcFaceResnet100Item:
    frame: np.ndarray
    landmarks_batch: np.ndarray
    bbox_batch: np.ndarray

    aligned_face_batch: np.ndarray = None

    face_embedding_batch: np.ndarray = None

    similarities: np.ndarray = None
    highest_target_similarity_index: int = None
    is_above_similarity_threshold: float = None
    matched_names: List[str] = None

    model_time: float = None
