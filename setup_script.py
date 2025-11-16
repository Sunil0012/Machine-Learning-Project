"""
Quick Setup Script
Run this first to set up the environment and verify installation
"""

import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✓ Python version OK")
    return True

def install_requirements():
    """Install required packages"""
    print_header("Installing Requirements")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def download_spacy_model():
    """Download spaCy model"""
    print_header("Downloading spaCy Model")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to download spaCy model")
        return False

def download_nltk_data():
    """Download NLTK data"""
    print_header("Downloading NLTK Data")
    
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def create_directories():
    """Create project directories"""
    print_header("Creating Directories")
    
    from config import Config
    
    try:
        Config.create_directories()
        print("✓ Directories created")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False

def check_data_files():
    """Check if data files exist"""
    print_header("Checking Data Files")
    
    required_files = [
        # "C:\Users\TL1\Desktop\Machine Learning\empathetic_dialogues_train.csv",
        # "C:\Users\TL1\Desktop\Machine Learning\empathetic_dialogues_validation.csv",
        # "C:\Users\TL1\Desktop\Machine Learning\empathetic_dialogues_test.csv"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ Found {file}")
        else:
            print(f"❌ Missing {file}")
            all_exist = False
    
    if not all_exist:
        print("\nPlease download the Empathetic Dialogues dataset:")
        print("https://github.com/facebookresearch/EmpatheticDialogues")
        print("Place the CSV files in data/raw/")
        return False
    
    return True

def verify_imports():
    """Verify all imports work"""
    print_header("Verifying Imports")
    
    imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("nltk", "NLTK"),
        ("spacy", "spaCy"),
        ("textblob", "TextBlob"),
        ("vaderSentiment", "VADER Sentiment"),
        ("empath", "Empath")
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Check GPU availability"""
    print_header("Checking GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ No GPU available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    print_header("Running Quick Test")
    
    try:
        from config import Config
        from bert_classifier import BERTEmotionClassifier
        
        # Test model creation
        model = BERTEmotionClassifier(n_classes=32)
        print("✓ Model creation successful")
        
        # Test emotion mappings
        emotion2id, id2emotion = Config.get_emotion_id_mappings()
        assert len(emotion2id) == 32
        print("✓ Emotion mappings OK")
        
        print("\nAll tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps"""
    print_header("Next Steps")
    
    print("""
1. Download the Empathetic Dialogues dataset:
   https://github.com/facebookresearch/EmpatheticDialogues
   
2. Place CSV files in data/raw/:
   - train.csv
   - valid.csv
   - test.csv

3. Run the complete pipeline:
   python main_pipeline.py --all
   
4. Or run step by step:
   python main_pipeline.py --preprocess
   python main_pipeline.py --bert
   python main_pipeline.py --gpt2
   python main_pipeline.py --evaluate
   
5. Start interactive chat:
   python main_pipeline.py --chat

For more details, see README.md
    """)

def main():
    """Main setup process"""
    print_header("Emotional Support Chatbot - Setup")
    print("This script will set up your environment")
    
    steps = [
        ("Python Version", check_python_version),
        ("Install Requirements", install_requirements),
        ("Download spaCy Model", download_spacy_model),
        ("Download NLTK Data", download_nltk_data),
        ("Create Directories", create_directories),
        ("Verify Imports", verify_imports),
        ("Check GPU", check_gpu),
        ("Check Data Files", check_data_files),
        ("Quick Test", run_quick_test)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print_header("Setup Summary")
    
    for step_name, result in results.items():
        status = "✓" if result else "❌"
        print(f"{status} {step_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        print_next_steps()
    else:
        print("\n⚠ Some steps failed. Please review the errors above.")
        print("You may need to:")
        print("  - Install missing dependencies manually")
        print("  - Download the dataset")
        print("  - Check your Python environment")

if __name__ == "__main__":
    main()