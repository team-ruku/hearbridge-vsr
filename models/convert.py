import torch
from unet import UNet

model = UNet(3, 2)
model.load_state_dict(torch.load("best_weights.pth"))
model.eval()
example = torch.rand(1, 3, 320, 480)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")
