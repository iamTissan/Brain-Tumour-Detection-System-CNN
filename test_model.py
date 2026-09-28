#!/usr/bin/env python3
"""
Brain Tumor Detection - Model Testing Script
Tests your trained model and provides detailed accuracy reports
"""

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms
import os
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

class BrainTumorTester:
    def __init__(self, model_path='best_model.pth'):
        """Initialize the tester with trained model"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.model = self.load_model()
        
        # Transform for testing (same as training validation)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the trained model"""
        print(f"\n{'='*70}")
        print("LOADING MODEL")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        try:
            # Create model architecture
            model = models.resnet50(pretrained=False)
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
                nn.Linear(256, 4)
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                accuracy = checkpoint.get('accuracy', 'N/A')
                print(f"✅ Model loaded successfully!")
                if isinstance(accuracy, float):
                    print(f"   Training accuracy: {accuracy:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Model loaded successfully!")
            
            model.to(self.device)
            model.eval()
            print(f"{'='*70}\n")
            return model
            
        except FileNotFoundError:
            print(f"❌ Model file '{self.model_path}' not found!")
            print("   Please train the model first using: python train_model_optimized.py")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            sys.exit(1)
    
    def predict_single(self, image_path):
        """Predict a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            pred_class = predicted.item()
            conf_score = confidence.item()
            all_probs = probabilities[0].cpu().numpy()
            
            return pred_class, conf_score, all_probs
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            return None, None, None
    
    def test_image(self, image_path):
        """Test a single image and display results"""
        print(f"\n{'='*70}")
        print(f"TESTING IMAGE: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        pred_class, confidence, all_probs = self.predict_single(image_path)
        
        if pred_class is None:
            return
        
        print(f"\n🎯 PREDICTION: {self.classes[pred_class]}")
        print(f"📊 CONFIDENCE: {confidence*100:.2f}%")
        
        # Confidence assessment
        if confidence > 0.9:
            print(f"✅ Assessment: Very High Confidence")
        elif confidence > 0.75:
            print(f"ℹ️  Assessment: High Confidence")
        elif confidence > 0.6:
            print(f"⚠️  Assessment: Moderate Confidence - Manual review recommended")
        else:
            print(f"❌ Assessment: Low Confidence - Needs manual review")
        
        print(f"\n📈 ALL CLASS PROBABILITIES:")
        for cls_name, prob in zip(self.classes, all_probs):
            bar = '█' * int(prob * 50)
            print(f"   {cls_name:15s} {prob*100:6.2f}% {bar}")
        
        print(f"{'='*70}\n")
    
    def test_folder(self, folder_path, true_label=None):
        """Test all images in a folder"""
        print(f"\n{'='*70}")
        print(f"TESTING FOLDER: {folder_path}")
        print(f"{'='*70}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder '{folder_path}' does not exist!")
            return
        
        for ext in image_extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ No images found in '{folder_path}'")
            return
        
        print(f"\nFound {len(image_files)} images")
        
        # Track results
        predictions = []
        confidences = []
        correct = 0
        
        # If true label provided, find its index
        true_class_idx = None
        if true_label:
            try:
                true_class_idx = self.classes.index(true_label)
            except ValueError:
                print(f"⚠️  Warning: Unknown class '{true_label}'")
        
        print(f"\nProcessing images...")
        for i, image_path in enumerate(image_files, 1):
            pred_class, confidence, _ = self.predict_single(str(image_path))
            
            if pred_class is not None:
                predictions.append(pred_class)
                confidences.append(confidence)
                
                if true_class_idx is not None and pred_class == true_class_idx:
                    correct += 1
                
                print(f"   [{i}/{len(image_files)}] {os.path.basename(image_path):30s} → "
                      f"{self.classes[pred_class]:15s} ({confidence*100:.1f}%)")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total images: {len(predictions)}")
        print(f"Average confidence: {np.mean(confidences)*100:.2f}%")
        
        if true_class_idx is not None:
            accuracy = (correct / len(predictions)) * 100
            print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(predictions)})")
        
        # Prediction distribution
        print(f"\nPrediction Distribution:")
        pred_counts = defaultdict(int)
        for pred in predictions:
            pred_counts[self.classes[pred]] += 1
        
        for cls_name in self.classes:
            count = pred_counts[cls_name]
            percentage = (count / len(predictions)) * 100
            bar = '█' * int(percentage / 2)
            print(f"   {cls_name:15s} {count:4d} ({percentage:5.1f}%) {bar}")
        
        print(f"{'='*70}\n")
    
    def test_dataset(self, dataset_path='dataset'):
        """Test entire dataset and generate accuracy report"""
        print(f"\n{'='*70}")
        print("TESTING ENTIRE DATASET")
        print(f"{'='*70}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"❌ Dataset folder '{dataset_path}' does not exist!")
            return
        
        # Find class folders
        class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
        
        if not class_folders:
            print(f"❌ No class folders found in '{dataset_path}'")
            return
        
        print(f"\nFound {len(class_folders)} classes")
        
        # Track results per class
        results = {}
        total_correct = 0
        total_images = 0
        
        for class_folder in class_folders:
            class_name = class_folder.name.capitalize()
            
            # Try to match to known classes
            matched_class = None
            for known_class in self.classes:
                if known_class.lower() in class_name.lower():
                    matched_class = known_class
                    break
            
            if not matched_class:
                print(f"⚠️  Skipping unknown class: {class_name}")
                continue
            
            print(f"\n📁 Testing class: {matched_class}")
            print("-" * 70)
            
            # Get all images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(class_folder.glob(f'*{ext}'))
                image_files.extend(class_folder.glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"   No images found")
                continue
            
            # Test images
            correct = 0
            confidences = []
            true_class_idx = self.classes.index(matched_class)
            
            for image_path in image_files:
                pred_class, confidence, _ = self.predict_single(str(image_path))
                
                if pred_class is not None:
                    confidences.append(confidence)
                    if pred_class == true_class_idx:
                        correct += 1
            
            # Calculate metrics
            accuracy = (correct / len(image_files)) * 100
            avg_confidence = np.mean(confidences) * 100 if confidences else 0
            
            results[matched_class] = {
                'total': len(image_files),
                'correct': correct,
                'accuracy': accuracy,
                'confidence': avg_confidence
            }
            
            total_correct += correct
            total_images += len(image_files)
            
            print(f"   Images: {len(image_files)}")
            print(f"   Correct: {correct}")
            print(f"   Accuracy: {accuracy:.2f}%")
            print(f"   Avg Confidence: {avg_confidence:.2f}%")
        
        # Overall summary
        print(f"\n{'='*70}")
        print("OVERALL RESULTS")
        print(f"{'='*70}")
        
        overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
        print(f"\n📊 Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"   Total Images: {total_images}")
        print(f"   Correct Predictions: {total_correct}")
        print(f"   Incorrect Predictions: {total_images - total_correct}")
        
        print(f"\n📈 Per-Class Results:")
        print(f"{'   Class':<20} {'Accuracy':>10} {'Images':>8} {'Avg Conf':>12}")
        print("   " + "-" * 60)
        
        for cls_name in self.classes:
            if cls_name in results:
                res = results[cls_name]
                print(f"   {cls_name:<20} {res['accuracy']:>9.2f}% "
                      f"{res['total']:>8} {res['confidence']:>11.2f}%")
        
        # Assessment
        print(f"\n🎯 Assessment:")
        if overall_accuracy >= 97:
            print(f"   ✅ EXCELLENT! Target accuracy achieved (97%+)")
        elif overall_accuracy >= 95:
            print(f"   ✅ VERY GOOD! Close to target (95-97%)")
        elif overall_accuracy >= 90:
            print(f"   ⚠️  GOOD but needs improvement (90-95%)")
        else:
            print(f"   ❌ NEEDS WORK (<90%)")
        
        # Identify weak classes
        weak_classes = [cls for cls, res in results.items() if res['accuracy'] < 90]
        if weak_classes:
            print(f"\n⚠️  Weak Classes (need attention):")
            for cls in weak_classes:
                print(f"   - {cls}: {results[cls]['accuracy']:.2f}%")
        
        print(f"{'='*70}\n")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("BRAIN TUMOR DETECTION - MODEL TESTER")
        print("="*70)
        print("\nUsage:")
        print("  python test_model.py <image_path>          - Test single image")
        print("  python test_model.py <folder_path> [class] - Test folder of images")
        print("  python test_model.py dataset               - Test entire dataset")
        print("\nExamples:")
        print("  python test_model.py test_image.jpg")
        print("  python test_model.py test_folder/ Glioma")
        print("  python test_model.py dataset/")
        print("="*70 + "\n")
        sys.exit(1)
    
    tester = BrainTumorTester()
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # Single image
        tester.test_image(path)
    elif os.path.isdir(path):
        if path.lower() in ['dataset', 'dataset/', './dataset', './dataset/']:
            # Full dataset test
            tester.test_dataset(path)
        else:
            # Folder test
            true_label = sys.argv[2] if len(sys.argv) > 2 else None
            tester.test_folder(path, true_label)
    else:
        print(f"❌ Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()