from dataclasses import dataclass
from typing import List

import numpy as np

from src.models.detection.item import DetectionItem


@dataclass
class ArcFaceResnet100Item(DetectionItem):
    frame: np.ndarray = None

    aligned_face_batch: np.ndarray = None

    face_embedding_batch: np.ndarray = None

    similarities: np.ndarray = None
    highest_target_similarity_index: int = None
    is_above_similarity_threshold: float = None
    matched_names: List[str] = None

    preprocess_time: float = None
    model_time: float = None
    postprocess_time: float = None
