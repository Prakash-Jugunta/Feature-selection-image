import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model import NetworkCIFAR as Network
import genotypes

# Argument parser
parser = argparse.ArgumentParser("malayalam-test")
parser.add_argument('--data', type=str, default='dataset/malayalam_chars', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
parser.add_argument('--arch', type=str, default='ADARTS', help='Architecture name')
parser.add_argument('--binary', action='store_true', help='Use binary classification (default: multi-class)')
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
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Determine number of classes
    num_classes = 1 if args.binary else len(test_data.classes)
    logging.info(f"Number of classes: {num_classes}")

    # Model setup
    genotype = eval(f"genotypes.{args.arch}")
    model = Network(C=16, num_classes=num_classes, layers=8, auxiliary=False, genotype=genotype)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load model weights
    utils.load(model, args.model_path)
    logging.info("Model loaded from %s", args.model_path)

    # Loss function
    criterion = nn.BCEWithLogitsLoss().to(device) if args.binary else nn.CrossEntropyLoss().to(device)

    # Run evaluation
    test_acc = infer(test_loader, model, criterion, args.binary, device)
    logging.info(f'Final Test Accuracy: {test_acc:.2f}%')

def infer(test_loader, model, criterion, binary, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)
            if binary:
                targets = targets.float()
            
            outputs = model(inputs)[0]  # Get main logits
            if binary:
                loss = criterion(outputs.squeeze(), targets)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else:
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
            
            test_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    logging.info(f'Test Loss: {avg_loss:.4f}')
    return accuracy

if __name__ == '__main__':
    main()