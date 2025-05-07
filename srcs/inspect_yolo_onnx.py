import onnx

path = "../models/onnx/yolopv2.onnx"
onnx_model = onnx.load(path)
print("Output shapes:")
for output in onnx_model.graph.output:
        print(f"{output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
