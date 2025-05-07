import torch

path = "../models/pt/yolopv2.pt"

model = torch.load(path, map_location='cpu', weights_only=False)  # Explicitly allow code execution
print(model) 

model = torch.jit.load(path, map_location="cpu")
print(model.graph)  # Shows input shape in the first few lines
