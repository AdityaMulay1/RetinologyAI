# Diabetic Retinopathy Detection AI
<img width="2239" height="1336" alt="image" src="https://github.com/user-attachments/assets/2856a7e6-8389-4ccd-910c-81a8cc6bdf41" />


Advanced AI application for detecting diabetic retinopathy using ResNet50 architecture.
<img width="3744" height="1206" alt="image" src="https://github.com/user-attachments/assets/11e9828f-faaa-41ef-86ac-d236cdd0f919" />

## Quick Start

```bash
git clone <your-repo-url>
cd Diabetic-Retinopathy-Detection
pip install -r requirements.txt
python setup_model.py
python enhanced_desktop_app_v2.py
```

## Features

- **85%+ Accuracy** with ResNet50 + ImageNet
- **5 Severity Levels** (Normal to Proliferative)
- **PDF Medical Reports** generation
- **Real-time Analysis** (2-5 seconds)
- **Professional UI** for medical use

## Classification Levels

| Level | Description | Action Required |
|-------|-------------|----------------|
| Normal | Healthy eye | Regular monitoring |
| Mild | Minor signs | 6-12 month follow-up |
| Moderate | Medical attention needed | 3-6 month specialist visit |
| Severe | Immediate treatment | Urgent medical care |
| Proliferative | Advanced stage | **EMERGENCY** specialist care |

## Usage

1. Run `python enhanced_desktop_app_v2.py`
2. Upload retinal fundus image
3. Click "AI Analysis"
4. View results and save PDF report

## Medical Disclaimer

This AI tool is for **screening purposes only**. Always consult qualified ophthalmologists for proper medical diagnosis and treatment decisions.

## Image Source
Google Images

## Technical Specs

- **Model**: ResNet50 + ImageNet Pre-trained
- **Input**: 512×512 retinal images
- **Output**: 5-class severity prediction
- **Requirements**: Python 3.8+, PyTorch, Tkinter
