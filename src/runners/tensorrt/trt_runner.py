from typing import List

import numpy as np

from src.runners.tensorrt.utils import allocate_buffers, free_buffers, do_inference_v2, load_engine


class TrtRunner:
    def __init__(self, trt_path):
        self.engine = load_engine(trt_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def infer(self, x: List[np.ndarray], copy_outputs=True):
        [np.copyto(inp.host, x_i.flatten()) for inp, x_i in zip(self.inputs, x)]

        outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)

        if copy_outputs:
            return [np.copy(out) for out in outputs]
        else:
            return outputs

    def __del__(self):
        free_buffers(self.inputs, self.outputs, self.stream)
