from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class YOLOv8nFaceItem:
    frame: np.ndarray

    blob: np.ndarray = None
    pad_h: int = None
    pad_w: int = None
    scale_h: int = None
    scale_w: int = None

    outputs: List[np.ndarray] = None

    det_bboxes: np.ndarray = None
    det_conf: np.ndarray = None
    det_classid: np.ndarray = None
    landmarks: np.ndarray = None