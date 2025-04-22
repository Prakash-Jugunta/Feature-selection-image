import os
import sys
import glob
from matplotlib import transforms
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from model import NetworkCIFAR as Network
import genotypes

# Argument parser
parser = argparse.ArgumentParser("malayalam-test")
parser.add_argument('--data', type=str, default='dataset', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
parser.add_argument('--arch', type=str, default='ADARTS', help='Architecture name')
args = parser.parse_args()

# Setup logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main():
    # Data transforms
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((86, 86)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load dataset
    test_data = ImageFolder(root=os.path.join(args.data, 'test'), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model setup
    genotype = eval(f"genotypes.{args.arch}")
    model = Network(C=36, num_classes=1, layers=20, auxiliary=False, genotype=genotype)
    
    # Modify input channels for grayscale
    model.stem = nn.Sequential(
        nn.Conv2d(1, 36*3, 3, padding=1, bias=False),
        nn.BatchNorm2d(36*3)
    )
    
    model = model.cuda()
    utils.load(model, args.model_path)
    logging.info("Model loaded from %s", args.model_path)

    # Loss function
    criterion = nn.BCEWithLogitsLoss().cuda()

    # Run evaluation
    test_acc = infer(test_loader, model, criterion)
    logging.info(f'Final Test Accuracy: {test_acc:.2f}%')

def infer(test_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            targets = labels.float().cuda()
            
            outputs = model(inputs)[0]  # Get main logits
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            correct += (predicted.squeeze() == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    main()
