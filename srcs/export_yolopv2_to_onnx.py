import torch

input_path = "../models/pt/yolopv2.pt"
output_path = "../models/onnx/yolopv2.onnx"

model = torch.jit.load(input_path , map_location="cpu")
model.eval()

# Input shape (adjust if needed!)
dummy_input = torch.randn(1, 3, 640, 640)  # [batch, channels, height, width]

# Export with output names matching the demo script
torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["pred_anchor", "seg", "ll"],  # Critical for TensorRT
        opset_version=12,
    dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                "pred_anchor": {0: "batch"},
                "seg": {0: "batch"},
                "ll": {0: "batch"},
    },
        do_constant_folding=True
)
