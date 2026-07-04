# End-to-End Deepfake Audio Detection Ensemble
**A Hybrid Temporal-Spectral Framework Using AASIST, ResNet-18, and Cross-Attention Fusion**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Domain](https://img.shields.io/badge/Domain-Audio%20Forensics%20%2F%20AI%20Safety-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview
Synthetic speech generation and voice conversion technologies have grown increasingly sophisticated, creating severe security vulnerabilities in biometric authentication and digital communication. This repository contains the official implementation of a dual-branch deep learning ensemble designed to detect deepfake audio across clean studio environments and degraded real-world transmission channels.

The framework extracts complementary acoustic features by processing raw time-domain waveforms through **AASIST** (Raw Audio Spoofing Detection System) and frequency-domain Mel-spectrograms through a customized **ResNet-18 with SimAM** (Simple Parameter-Free Attention Module). The feature embeddings from both branches are integrated using a **Cross-Attention Fusion Layer** to classify spoken utterances as either real human speech (*bonafide*) or artificial generation (*spoof*).

---

## Key Features
* **Dual-Domain Feature Extraction:** Processes both 1d raw audio waveforms and 2d Mel-spectrogram representations simultaneously to capture temporal artifacts and spectral anomalies.
* **Cross-Attention Feature Fusion:** Replaces standard concatenation with a multi-head attention mechanism that dynamically weights temporal and spectral embeddings based on their diagnostic reliability.
* **Resilient Preprocessing Pipeline:** Incorporates a bulletproof audio loading mechanism using `soundfile` and `pydub` with automated resampling to 16 kHz, Voice Activity Detection (VAD) normalization, and signal pre-emphasis ($\alpha = 0.97$).
* **Comprehensive Evaluation Suite:** Calculates industry-standard biometric verification metrics including Equal Error Rate (EER) and minimum tandem Decision Cost Function (min t-DCF).
* **In-The-Wild Testing Pipeline:** Features dedicated scripts to evaluate model generalization against custom recorded phone calls and web-captured AI voice previews across 9 distinct codec degradation formats.
* **Automated Visualizations:** Generates high-resolution publication-ready graphs including Score Distribution plots, Receiver Operating Characteristic (ROC) curves, Detection Error Tradeoff (DET) curves, Normalized DCF curves, and Confusion Matrices.

---

## System Architecture

### 1. Preprocessing Pipeline
All incoming audio signals are standardized to ensure consistent feature extraction:
* **Sampling Rate:** 16,000 Hz.
* **Target Sequence Length:** 64,600 samples ($4.0375$ seconds). Shorter utterances are reflect-padded, while longer sequences are center-cropped.
* **Signal Pre-emphasis:** Applied using the first-order difference equation:
  $$y[n] = x[n] - 0.97 \cdot x[n-1]$$
* **Mel-Spectrogram Generation:** Computed using a 512-point Fast Fourier Transform (FFT), a hop length of 160 samples, and 80 Mel frequency bins scaled to power decibels (top dB = 80).

### 2. Model Branches
* **Branch A (Temporal Domain):** An **AASIST** backbone featuring graph attention networks (GATs) and heterogeneous graph convolution layers. It processes raw waveforms to identify subtle phase and waveform continuity distortions caused by neural vocoders.
* **Branch B (Spectral Domain):** A **ResNet-18** architecture enhanced with **SimAM** 3D attention modules. It processes 80-channel Mel-spectrograms to detect acoustic band smearing, missing harmonic frequencies, and codec compression artifacts.
* **Cross-Attention Fusion Head:** Projects the 104-dimensional AASIST embedding and the 512-dimensional ResNet embedding into a shared 256-dimensional latent space. A 2-layer Transformer encoder applies self-attention and cross-attention before routing the combined token to the final classification layer.

---

## Repository Structure
```text
├── src/
│   ├── data/
│   │   └── dataset.py             # Data loader and augmentation pipeline
│   ├── models/
│   │   ├── aasist.py              # AASIST model architecture
│   │   └── resnet_simam.py        # ResNet-18 with SimAM attention module
│   └── ur_ffl/
│       ├── actuator.py            # On-the-fly signal degradation simulators
│       ├── selector.py            # Feature selection and routing logic
│       └── sensor.py              # Cross-attention fusion layer implementation
├── extract_samples.py             # Script to extract and convert audio samples to MP4/Spectrograms
├── evaluate_ensemble.py           # Main evaluation script for ASVspoof challenge datasets
├── inthewild_evaluation.py        # Standalone evaluation script for custom recorded datasets
├── filter_DF_dataset.py           # Utility script to analyze degradation distributions
├── requirements.txt               # List of Python project dependencies
└── README.md                      # Project documentation
