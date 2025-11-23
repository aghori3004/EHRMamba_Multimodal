import torch
import mamba_ssm
from mamba_ssm import Mamba

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Mamba: {mamba_ssm.__version__}")

# Quick GPU Test
try:
    device = "cuda"
    model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).to(device)
    x = torch.randn(2, 64, 16).to(device)
    y = model(x)
    print("✅ Success: Mamba is running on your RTX 4060!")
except Exception as e:
    print(f"❌ Error: {e}")