#!/usr/bin/env python3
"""
One-click installer for Diabetic Retinopathy Detection AI
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    print("🚀 Installing Diabetic Retinopathy Detection AI...")
    
    try:
        # Install requirements
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Setup model
        print("🤖 Setting up AI model...")
        subprocess.check_call([sys.executable, "setup_model.py"])
        
        print("\n✅ Installation Complete!")
        print("🚀 Run: python enhanced_desktop_app_v2.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()