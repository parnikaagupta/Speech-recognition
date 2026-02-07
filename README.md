# Speech-recognition
AI CLUB TASK 2
# Speech Emotion Recognition 🎤

NAME : PARNIKA GUPTA 
ID : 2025B4PS1105P

## Overview
This project implements a Speech Emotion Recognition (SER) system using
MFCC-based audio features and a CNN + BiLSTM neural network.

## Dataset
- RAVDESS Emotional Speech Dataset
- 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised

## Features
- MFCC + Delta + Delta-Delta
- Z-score normalization
- Data augmentation (noise, pitch shift, time stretch)

## Model Architecture
- CNN layers for spatial feature extraction
- BiLSTM for temporal modeling
- Fully connected classifier

## Training Details
- Optimizer: Adam
- Loss: CrossEntropyLoss (with class weighting)
- Epochs: 20–30
- Batch size: 32

## Performance
- Test Accuracy: ~71–80% (varies with augmentation)

## How to Run Inference
```bash
pip install -r requirements.txt
python inference.py path_to_audio.wav
