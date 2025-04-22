import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import logging
from model import NetworkCIFAR as Network
import genotypes
import utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%I:%M:%S %p')

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Predict Malayalam character from a single image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='results-20250414-112908/best_weights.pt', 
                        help='Path to the trained model weights')
    args = parser.parse_args()

    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((86, 86)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load and preprocess the image
    try:
        image = Image.open(args.image_path).convert('L')  # Convert to grayscale
        image = transform(image).unsqueeze(0)  # Add batch dimension
        logging.info(f"Image loaded and preprocessed from {args.image_path}")
    except Exception as e:
        logging.error(f"Failed to load image: {e}")
        return

    # Load class names (assuming test directory structure from training)
    from torchvision.datasets import ImageFolder
    test_data = ImageFolder(root='./dataset/malayalam_chars/test')
    class_names = test_data.classes
    num_classes = len(class_names)
    logging.info(f"Detected {num_classes} classes: {class_names}")

    # Initialize the model
    genotype = getattr(genotypes, 'ADARTS')
    model = Network(C=16, num_classes=num_classes, layers=8, auxiliary=False, genotype=genotype)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load the trained weights
    try:
        utils.load(model, args.model_path)
        logging.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Perform prediction
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)[0]  # Get the main logits
        _, predicted = output.max(1)
        predicted_class = class_names[predicted.item()]
        probabilities = torch.softmax(output, dim=1)
        top_prob, top_class_idx = probabilities.max(1)
        confidence = top_prob.item()

    # Log the prediction
    logging.info(f"Predicted character: {predicted_class} (Confidence: {confidence:.2f})")
    logging.info(f"Class probabilities: {dict(zip(class_names, probabilities.cpu().numpy()[0]))}")

if __name__ == '__main__':
    main()