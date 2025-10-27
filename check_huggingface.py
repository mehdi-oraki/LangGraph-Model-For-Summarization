"""
Hugging Face Installation Verification Script
Run this to ensure your Hugging Face setup is working correctly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def check_huggingface_setup():
    """Verify Hugging Face installation and model availability"""
    print("🔍 Checking Hugging Face Setup...")
    
    # Check PyTorch installation
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    # Check if we can load a small model
    try:
        print("📦 Testing model loading...")
        model_name = "google/flan-t5-small"  # Small, efficient model
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer loaded: {model_name}")
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"✅ Model loaded: {model_name}")
        
        # Test inference
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test inference successful: '{test_text}' -> '{result}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup check: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Ensure you have internet connection for first-time model download")
        print("2. Check if you have sufficient disk space")
        print("3. Verify all dependencies are installed correctly")
        return False

if __name__ == "__main__":
    success = check_huggingface_setup()
    if success:
        print("\n🎉 Hugging Face setup is ready!")
    else:
        print("\n⚠️ Please fix the issues above before proceeding.")