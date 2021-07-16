import argparse
import os
from util import get_device, image_loader
import numpy
import torch
from torchvision import transforms

from run import train, get_config
from eval import evaluate
from PIL import Image

parser = argparse.ArgumentParser(description='MNIST Example')
parser.add_argument('--ckpf', help="path to model checkpoint file (to continue training)")
parser.add_argument('--train', action='store_true', help='training a ConvNet model on MNIST dataset')
parser.add_argument('--evaluate', action='store_true', help='evaluate a [pre]trained model')
parser.add_argument('--image_path', help='image to evaluate')


args = parser.parse_args()
config = get_config()
device = get_device()
tfs = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

if args.train:
    train(config)

elif args.evaluate:
    input = image_loader(args.image_path)
    output = evaluate(args.ckpf, input)
    output = output.cpu()
    label = torch.max(output, 1)[1].data.numpy().squeeze()
    print(f"Predicted Label - {label}")
