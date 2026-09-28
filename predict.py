from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch


def predict_single(model, image, device, transform=None, labels=None):
    """
    Predict single image with the trained model
    
    Args:
        model: Trained PyTorch model
        image: Path to image or PIL Image
        device: torch device (cpu or cuda)
        transform: Optional transform (will use default if None)
        labels: Optional list of class labels
    
    Returns:
        pred: Predicted class (label if provided, else index)
        pred_prob: Prediction probability
    """
    
    # Load image if path is provided
    if isinstance(image, str):
        original_image = Image.open(image).convert('RGB')
    else:
        original_image = image.convert('RGB')

    # Default transform matching training
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    img = transform(original_image).to(device)

    model.eval()
    with torch.no_grad():
        model.to(device)
        pred_logits = model(img.unsqueeze(dim=0))

        # Get prediction and probability
        pred_probs = torch.softmax(pred_logits, dim=-1)
        pred = pred_probs.argmax(dim=-1)
        pred_prob = pred_probs.max().item()
        
        # Get all class probabilities
        all_probs = pred_probs[0].cpu().numpy()

        if labels:
            pred_label = labels[pred.item()]
            return pred_label, pred_prob, all_probs
        
        return pred.item(), pred_prob, all_probs


def predict_batch(model, image_paths, device, labels=None):
    """
    Predict multiple images at once
    
    Args:
        model: Trained PyTorch model
        image_paths: List of image paths
        device: torch device
        labels: Optional list of class labels
    
    Returns:
        predictions: List of (prediction, probability) tuples
    """
    predictions = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=-1)
            pred = probs.argmax(dim=-1).item()
            prob = probs.max().item()
            
            if labels:
                pred = labels[pred]
            
            predictions.append((pred, prob))
    
    return predictions


def visualize_prediction(image_path, prediction, probability, labels=None):
    """
    Visualize image with prediction
    """
    img = Image.open(image_path)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    
    title = f"Prediction: {prediction}\nConfidence: {probability*100:.2f}%"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()