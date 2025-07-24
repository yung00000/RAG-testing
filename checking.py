import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # For NVIDIA
print(f"ROCm/HIP available: {torch.version.hip}")     # Only if ROCm is installed
print(f"Running on CPU: {not torch.cuda.is_available()}")