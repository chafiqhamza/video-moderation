#!/usr/bin/env python3
"""
🚀 MAIN PROJECT LAUNCHER
Launches the AI system with proper environment setup
"""

import os
import sys
import subprocess

def setup_python_path():
    """Setup Python path for imports"""
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    ai_models_path = os.path.join(backend_path, 'ai_models')
    
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    if ai_models_path not in sys.path:
        sys.path.insert(0, ai_models_path)
    
    print(f"✅ Added to Python path: {backend_path}")
    print(f"✅ Added to Python path: {ai_models_path}")

def test_imports():
    """Test essential imports"""
    print("🔍 Testing essential imports...")
    
    try:
        # Test AI models import
        from ai_models.youtube_policy_classifier import YouTubePolicyClassifier
        print("✅ YouTube Policy Classifier import successful")
        
        # Test basic libraries
        import cv2, numpy, PIL, easyocr
        print("✅ Basic libraries available")
        
        # Test Flask
        from flask import Flask
        print("✅ Flask available")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def start_backend():
    """Start the backend server"""
    print("🚀 Starting backend server...")
    
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    
    try:
        subprocess.run([sys.executable, 'main.py'], check=True)
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    """Main launcher function"""
    print("🎯 AI SYSTEM LAUNCHER")
    print("="*40)
    
    # Setup environment
    setup_python_path()
    
    # Test imports
    if not test_imports():
        print("❌ Import tests failed. Please install dependencies.")
        return
    
    print("✅ All imports successful")
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main()
