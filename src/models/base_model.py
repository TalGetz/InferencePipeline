from src.runners.runner import Runner


class BaseModel:
    def __init__(self, model_path, output_shapes):
        self.runner = Runner(model_path)
        self.output_shapes = output_shapes

    def infer(self, x, copy_outputs=True):
        outputs = self.runner.infer(x, copy_outputs=copy_outputs)
        assert len(outputs) == len(self.output_shapes)
        return [output.reshape(output_shape) for output, output_shape in zip(outputs, self.output_shapes)]
