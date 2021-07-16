from model import ConvNet
from run import get_device
import torch
from torchvision import transforms

def evaluate(ckpf, input):
    print(f"INput shape - {input.shape}")
    device = get_device()
    input = input
    model = ConvNet()
    model = model.to(device)
    model_state_dict = torch.load(ckpf)
    model.load_state_dict(model_state_dict)
    model.eval()
    label = model(input)
    return label
    
