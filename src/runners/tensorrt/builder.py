import enum
from pathlib import Path

import tensorrt as trt
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class Precisions(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"


def build_engine(input_onnx_path, quantize, output_trt_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if quantize == "fp16":
        config.flags |= 1 << int(trt.BuilderFlag.FP16)

    config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
    config.default_device_type = trt.DeviceType.GPU

    with open(input_onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise ValueError(f"Could not read onnx file {input_onnx_path}")

    # profile = builder.create_optimization_profile()
    # profile.set_shape("input", 1, 1, 1)
    # config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)

    with open(output_trt_path, 'wb') as f:
        f.write(bytearray(engine))

    return engine


def main(input_onnx_path, precision, output_trt_path):
    precision_values = [p.value for p in Precisions]
    if precision not in precision_values:
        raise ValueError(f"Precision of '{precision}' is not in: {precision_values}")

    if output_trt_path is None:
        output_trt_path = Path(input_onnx_path)
        output_trt_path = output_trt_path.parent / (output_trt_path.stem + ".trt")

    if os.path.exists(output_trt_path):
        raise FileExistsError(f"TRT Output file already exists: {output_trt_path}")

    build_engine(input_onnx_path=input_onnx_path, quantize=precision, output_trt_path=output_trt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_onnx_path", required=True)
    parser.add_argument("--output_trt_path", required=False, default=None)
    parser.add_argument("--precision", required=False, default="fp16")
    args = parser.parse_args()
    main(args.input_onnx_path, args.precision, args.output_trt_path)
