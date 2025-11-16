#!/usr/bin/env python3
"""
Model Setup Script for Diabetic Retinopathy Detection
Creates the enhanced model file for the application
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def create_enhanced_model():
    """Create and save the enhanced ResNet50 model"""
    print("🚀 Creating Enhanced ResNet50 Model...")
    
    # Create ResNet50 model
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes for DR severity
    
    # Save model
    model_path = 'enhanced_diabetic_retinopathy_model.pth'
    torch.save(model.state_dict(), model_path)
    
    print(f"✅ Enhanced model saved: {model_path}")
    print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    print("🎯 Ready for enhanced_desktop_app_v2.py")

if __name__ == "__main__":
    create_enhanced_model()