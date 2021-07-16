import torch
from PIL import Image
from torchvision import transforms

def get_device():
    if torch.cuda.is_available():
        print("GPU Available")
    else:
        print("Running on CPU")
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    return device

def image_loader(image_path):
    device = get_device()
    loader = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.CenterCrop((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])  
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    print(f"Input shape {image.shape}")
    return image.to(device)
