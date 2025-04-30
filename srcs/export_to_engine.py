import tensorrt as trt
import os
def build_engine_from_onnx(onnx_file_path, engine_file_path, precision='FP16', dynamic=False,
                           opt_batch_size=1, min_shape=(1, 3, 640, 640),
                           opt_shape=(1, 3, 640, 640), max_shape=(1, 3, 640, 640)):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # Parse o modelo ONNX
    with open(onnx_file_path, 'rb') as model:
        parser = trt.OnnxParser(network, TRT_LOGGER)
        if not parser.parse(model.read()):
            print("WARNING: parsing ONNX:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    # Configura precisão
    if precision == 'FP16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("WARNING: FP16 not supported, using FP32.")
    elif precision == 'INT8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("WARNING: INT8 not supported, using FP32.")
        else:
            print("WARNING: INT8 not supported, using FP32.")
    # Configura batch size dinâmico, se necessário
    if dynamic:
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    else:
        network.get_input(0).shape = opt_shape  # batch size fixo
    # Constrói o motor
    engine = builder.build_engine(network, config)
    if engine is None:
        print("ERROR: building the engine.")
        return False
    # Serializa o motor
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    return True
if __name__ == '__main__':
    onnx_file = 'yolov5n.onnx'
    engine_file = 'yolov5n_v30.engine'
    if build_engine_from_onnx(onnx_file, engine_file, precision='FP16'):
        print(f"SUCESS: TensorRT engine succefully build in '{engine_file}'.")
    else:
        print("FAILED: Constructing TensorRT engine.")
