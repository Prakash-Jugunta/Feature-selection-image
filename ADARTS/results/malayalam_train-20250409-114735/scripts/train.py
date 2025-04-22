import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import NetworkCIFAR as Network
import genotypes


# Argument parser
parser = argparse.ArgumentParser("malayalam")
parser.add_argument('--data', type=str, default='dataset', help='Dataset path')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save', type=str, default='results')
parser.add_argument('--arch', type=str, default='ADARTS')
args = parser.parse_args()

# Setup logging
args.save = f'{args.save}-{time.strftime("%Y%m%d-%H%M%S")}'
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    # Data transforms for Malayalam characters
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((86, 86)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((86, 86)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load datasets
    train_data = ImageFolder(root=os.path.join(args.data, 'train'), transform=train_transform)
    test_data = ImageFolder(root=os.path.join(args.data, 'test'), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Model configuration for single character
    genotype = eval(f"genotypes.{args.arch}")
    model = Network(C=36, num_classes=1, layers=20, auxiliary=False, genotype=genotype)
    
    # Modify input channels for grayscale
    model.stem = nn.Sequential(
        nn.Conv2d(1, 36*3, 3, padding=1, bias=False),
        nn.BatchNorm2d(36*3)
    )
    model = model.cuda()

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.cuda()
            targets = labels.float().cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)[0]  # Get main logits
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted.squeeze() == targets).sum().item()
            total += targets.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.cuda()
                targets = labels.float().cuda()
                
                outputs = model(inputs)[0]
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted.squeeze() == targets).sum().item()
                val_total += targets.size(0)

        # Log results
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        logging.info(f'Epoch {epoch+1:03d}: '
                     f'Train Loss: {train_loss/total:.4f} | Acc: {train_acc:.2f}% '
                     f'Val Loss: {val_loss/val_total:.4f} | Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            utils.save(model, os.path.join(args.save, 'best_weights.pt'))

    logging.info(f'Final Best Validation Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()


