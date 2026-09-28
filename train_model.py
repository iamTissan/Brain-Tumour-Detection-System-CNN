import torch
import torch.nn as nn
import torchvision.models as models
from engine import training_loop, FocalLoss
from preprocess import prepare_data

def set_device():
    """Set device to CUDA if available, else CPU"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


def create_model(num_classes=4, pretrained=True):
    """
    Create optimized ResNet50 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pretrained weights
    """
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    # Load ResNet50
    if pretrained:
        try:
            # Try loading custom weights first
            print("Attempting to load custom ResNet50 weights...")
            model = models.resnet50(pretrained=False)
            state_dict = torch.load("resnet50.pth", map_location="cpu")
            model.load_state_dict(state_dict)
            print("✅ Loaded custom ResNet50 weights")
        except:
            # Fall back to torchvision pretrained
            print("Custom weights not found. Using torchvision pretrained weights...")
            model = models.resnet50(pretrained=True)
            print("✅ Loaded torchvision pretrained weights")
    else:
        model = models.resnet50(pretrained=False)
        print("⚠️  Training from scratch (not recommended)")
    
    # Freeze early layers, fine-tune deeper layers
    for name, param in model.named_parameters():
        if "layer1" in name:
            param.requires_grad = False
        elif "layer2" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Replace final classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen parameters: {total-trainable:,} ({100*(total-trainable)/total:.1f}%)")
    print("="*70 + "\n")
    
    return model


def main():
    """Main training function"""
    
    # Set device
    device = set_device()
    
    # Prepare data
    print("\nPreparing data...")
    train_loader, val_loader, classes = prepare_data()
    num_classes = len(classes)
    
    print(f"\nClasses: {classes}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = create_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Loss function - Focal Loss
    criterion = FocalLoss(alpha=1, gamma=2)
    print(f"\nLoss Function: Focal Loss (better for hard examples)")
    
    # Optimizer - AdamW
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    print(f"Optimizer: AdamW (lr=0.001, weight_decay=0.01)")
    
    # Learning rate scheduler - OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    print(f"Scheduler: OneCycleLR (for optimal convergence)")
    
    # Train model
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    trained_model, best_acc = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=30,
        device=device,
        save_path="best_model.pth",
        early_stopping_patience=7,
        use_amp=False  # Disabled for CPU
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: best_model.pth")
    print("\nTo use the model, run:")
    print("  streamlit run app_optimized.py")
    print("="*70 + "\n")
    
    return trained_model, best_acc


if __name__ == "__main__":
    # This block only runs when script is executed directly
    trained_model, accuracy = main()
    
    if accuracy >= 97.0:
        print(f"🎉 SUCCESS! Achieved {accuracy:.2f}% accuracy!")
    elif accuracy >= 95.0:
        print(f"✅ Good! Achieved {accuracy:.2f}% accuracy")
        print("💡 Tips to reach 97%+:")
        print("   - Collect more training data")
        print("   - Ensure balanced classes")
        print("   - Train for more epochs (40-50)")
    else:
        print(f"⚠️  Achieved {accuracy:.2f}% accuracy")
        print("💡 Troubleshooting:")
        print("   - Check dataset quality and balance")
        print("   - Verify images are correct format")
        print("   - Try training longer")