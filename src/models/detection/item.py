from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class DetectionItem:
    frame: np.ndarray

    blob: np.ndarray = None
    pad_h: int = None
    pad_w: int = None
    scale_h: int = None
    scale_w: int = None

    outputs: List[np.ndarray] = None

    detection_bboxes: np.ndarray = None
    detection_confidences: np.ndarray = None
    detection_class_id: np.ndarray = None
    landmarks: np.ndarray = None

    preprocess_time: float = None
    model_time: float = None
    postprocess_time: float = None
