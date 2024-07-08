from pathlib import Path

from src.runners.tensorrt.trt_runner import TrtRunner


class Runner:
    def __init__(self, model_path):
        extension = Path(model_path).suffix
        if extension == "trt":
            self.runner = TrtRunner(model_path)
        elif extension == "onnx":
            raise NotImplementedError()
        elif extension == "pt":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def infer(self, x, copy_outputs=True):
        return self.runner.infer(x, copy_outputs=copy_outputs)