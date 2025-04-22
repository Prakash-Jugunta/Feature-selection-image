import os
import sys
import time
import glob
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
parser = argparse.ArgumentParser("telugu")
parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--save', type=str, default='results', help='Directory to save results')
parser.add_argument('--arch', type=str, default='ADARTS', help='Architecture name')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='Drop path probability')
args = parser.parse_args()

# Setup logging and CUDA
args.save = f'{args.save}-{time.strftime("%Y%m%d-%H%M%S")}'
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    train_transform, test_transform = utils._data_transforms_malayalam()

    train_data = ImageFolder(root=os.path.join(args.data, 'train'), transform=train_transform)
    test_data = ImageFolder(root=os.path.join(args.data, 'test'), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Get number of classes dynamically
    num_classes = len(train_data.classes)
    
    # Simplified model config
    genotype = eval(f"genotypes.{args.arch}")
    model = Network(C=16, num_classes=num_classes, layers=8, auxiliary=False, genotype=genotype)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_acc = 0.0

    for epoch in range(args.epochs):
        try:
            model.train()
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)[0]
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100 * correct / total

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logging.warning("WARNING: CUDA out of memory during training loop")
                torch.cuda.empty_cache()
                continue
            else:
                raise

        # Validation
        try:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = labels.to(device, non_blocking=True)

                    outputs = model(inputs)[0]
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_acc = 100 * val_correct / val_total

            if val_acc > best_acc:
                best_acc = val_acc
                utils.save(model, os.path.join(args.save, 'best_weights.pt'))

            logging.info(f'Epoch {epoch+1:03d}: '
                         f'Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logging.warning("WARNING: CUDA out of memory during validation loop")
                torch.cuda.empty_cache()
                continue
            else:
                raise

        torch.cuda.empty_cache()

    logging.info(f'Final Best Validation Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()