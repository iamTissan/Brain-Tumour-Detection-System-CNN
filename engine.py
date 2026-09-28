import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples
    Helps improve accuracy on difficult cases
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def training_loop(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer,
    scheduler=None,
    n_epochs=30, 
    device='cpu', 
    save_path="best_model.pth",
    early_stopping_patience=7,
    use_amp=True):
    """
    Optimized training loop for maximum accuracy
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        n_epochs: Number of epochs
        device: Device to train on
        save_path: Path to save best model
        early_stopping_patience: Epochs to wait before early stopping
        use_amp: Use automatic mixed precision (faster on modern GPUs)
    """
    
    best_acc = 0.0
    best_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Mixed precision: {use_amp and torch.cuda.is_available()}")
    print(f"{'='*70}\n")
    
    for epoch in range(n_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{n_epochs}")
        print(f"{'='*70}")
        
        # ==================== TRAINING PHASE ====================
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * (predicted == labels).sum().item() / labels.size(0):.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * running_correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"\n📊 Training Results:")
        print(f"   Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
        # ==================== VALIDATION PHASE ====================
        val_loss, val_acc, class_accs = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"\n🎯 Validation Results:")
        print(f"   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
        print(f"\n   Per-class Accuracy:")
        for cls_name, cls_acc in class_accs.items():
            print(f"      {cls_name}: {cls_acc:.2f}%")
        
        # ==================== LEARNING RATE SCHEDULING ====================
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n📉 Learning Rate: {current_lr:.6f}")
        
        # ==================== MODEL CHECKPOINTING ====================
        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            epochs_without_improvement = 0
            
            # Save model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'loss': val_loss,
                'class_accuracies': class_accs
            }, save_path)
            
            print(f"\n✅ NEW BEST MODEL SAVED!")
            print(f"   Accuracy: {best_acc:.2f}%")
            print(f"   Loss: {best_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"\n⏳ No improvement for {epochs_without_improvement} epoch(s)")
            print(f"   Best accuracy so far: {best_acc:.2f}%")
        
        # ==================== EARLY STOPPING ====================
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*70}")
            print(f"⚠️  EARLY STOPPING TRIGGERED")
            print(f"{'='*70}")
            print(f"No improvement for {early_stopping_patience} consecutive epochs")
            print(f"Best validation accuracy: {best_acc:.2f}%")
            break
    
    # ==================== TRAINING COMPLETE ====================
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return model, best_acc


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on validation/test set
    Returns loss, accuracy, and per-class accuracies
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                class_total[label_item] += 1
                if label == pred:
                    class_correct[label_item] += 1
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracies
    class_accuracies = {}
    for class_idx in sorted(class_correct.keys()):
        class_acc = 100.0 * class_correct[class_idx] / class_total[class_idx]
        class_accuracies[f"Class_{class_idx}"] = class_acc
    
    return avg_loss, accuracy, class_accuracies