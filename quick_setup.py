#!/usr/bin/env python3
"""
Brain Tumor Detection - One-Click Setup & Training
Automates the entire setup and training process
"""

import os
import sys
import subprocess
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def print_step(step_num, text):
    """Print step number"""
    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {text}")
    print(f"{'='*70}\n")

def check_python():
    """Check Python version"""
    print_step(1, "Checking Python Installation")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required!")
        sys.exit(1)
    
    print("✅ Python version OK")

def install_dependencies():
    """Install required packages"""
    print_step(2, "Installing Dependencies")
    
    packages = [
        'torch',
        'torchvision', 
        'pillow',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'streamlit',
        'tqdm'
    ]
    
    print("Installing packages:")
    for pkg in packages:
        print(f"  - {pkg}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet'
        ] + packages)
        print("\n✅ All dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies")
        print("Try manually: pip install torch torchvision pillow numpy scikit-learn matplotlib streamlit tqdm")
        sys.exit(1)

def check_dataset():
    """Check if dataset exists"""
    print_step(3, "Checking Dataset")
    
    if not os.path.exists('dataset'):
        print("❌ Dataset folder not found!")
        print("\n Creating dataset structure...")
        
        os.makedirs('dataset/glioma', exist_ok=True)
        os.makedirs('dataset/meningioma', exist_ok=True)
        os.makedirs('dataset/notumor', exist_ok=True)
        os.makedirs('dataset/pituitary', exist_ok=True)
        
        print("\n⚠️  Created dataset folders:")
        print("   dataset/glioma/")
        print("   dataset/meningioma/")
        print("   dataset/notumor/")
        print("   dataset/pituitary/")
        print("\n📌 Please add your MRI images to these folders, then run this script again.")
        sys.exit(0)
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    total_images = 0
    class_counts = {}
    
    print("Dataset structure found. Checking images...\n")
    
    for cls in classes:
        cls_path = os.path.join('dataset', cls)
        if os.path.exists(cls_path):
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            class_counts[cls] = count
            total_images += count
            
            status = "✅" if count >= 50 else "⚠️ "
            print(f"   {status} {cls:15s}: {count:4d} images")
    
    print(f"\n   Total: {total_images} images")
    
    if total_images == 0:
        print("\n❌ No images found in dataset folders!")
        print("   Please add your MRI images to the class folders.")
        sys.exit(1)
    
    if total_images < 200:
        print("\n⚠️  Warning: Dataset is small (< 200 images)")
        print("   Recommended: At least 200 images (50+ per class)")
        print("   Current dataset may not achieve 97%+ accuracy")
        
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    print("\n✅ Dataset OK!")

def check_gpu():
    """Check if GPU is available"""
    print_step(4, "Checking GPU Availability")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("ℹ️  No GPU detected. Training will use CPU (slower)")
            print("   Expected training time: 60-90 minutes")
    except ImportError:
        print("⚠️  Cannot check GPU (torch not installed yet)")

def train_model():
    """Train the model"""
    print_step(5, "Training Model")
    
    if not os.path.exists('train_model.py'):
        print("❌ train_model.py not found!")
        print("   Please ensure all files are in the current directory.")
        sys.exit(1)
    
    print("Starting training...")
    print("This will take 15-30 minutes with GPU, 60-90 minutes with CPU")
    print("You can stop anytime with Ctrl+C\n")
    
    time.sleep(2)
    
    try:
        subprocess.check_call([sys.executable, 'train_model.py'])
        print("\n✅ Training completed!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Training failed!")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return False

def check_model():
    """Check if model was trained"""
    print_step(6, "Checking Trained Model")
    
    if os.path.exists('best_model.pth'):
        print("✅ Model file found: best_model.pth")
        
        try:
            import torch
            checkpoint = torch.load('best_model.pth', map_location='cpu')
            if isinstance(checkpoint, dict) and 'accuracy' in checkpoint:
                acc = checkpoint['accuracy']
                print(f"   Accuracy: {acc:.2f}%")
                
                if acc >= 97:
                    print(f"   🎉 EXCELLENT! Target achieved!")
                elif acc >= 95:
                    print(f"   ✅ Very good! Close to target")
                elif acc >= 90:
                    print(f"   ⚠️  Good but needs improvement")
                else:
                    print(f"   ❌ Needs work - consider more data or tuning")
        except:
            print("   Model found but couldn't read accuracy")
        
        return True
    else:
        print("❌ Model file not found!")
        print("   Training may have failed or been interrupted")
        return False

def run_web_app():
    """Run the web application"""
    print_step(7, "Launching Web Application")
    
    if not os.path.exists('app.py'):
        print("❌ app.py not found!")
        sys.exit(1)
    
    print("Starting Streamlit web app...")
    print("The app will open in your browser automatically")
    print("If not, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the app\n")
    
    time.sleep(2)
    
    try:
        # FIXED HERE ↓↓↓↓↓
        subprocess.check_call([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n\n✅ Web app stopped")

def main():
    """Main setup flow"""
    print_header("🧠 BRAIN TUMOR DETECTION - AUTO SETUP")
    print("This script will:")
    print("  1. Check Python version")
    print("  2. Install dependencies")
    print("  3. Check dataset")
    print("  4. Check GPU availability")
    print("  5. Train the model")
    print("  6. Launch web app")
    print("\nThis may take 20-90 minutes depending on your hardware.\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled")
        sys.exit(0)
    
    check_python()
    install_dependencies()
    check_dataset()
    check_gpu()
    
    print("\n" + "="*70)
    response = input("Ready to start training? (y/n): ")
    if response.lower() != 'y':
        print("\nSetup complete. To train later, run:")
        print("  python train_model.py")
        sys.exit(0)
    
    training_success = train_model()
    
    if training_success and check_model():
        print("\n" + "="*70)
        response = input("Launch web app now? (y/n): ")
        if response.lower() == 'y':
            run_web_app()
        else:
            print("\nTo launch web app later, run:")
            print("  streamlit run app.py")
    else:
        print("\n⚠️  Training incomplete. Please check errors above.")
        print("\nTo retry training:")
        print("  python train_model.py")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
