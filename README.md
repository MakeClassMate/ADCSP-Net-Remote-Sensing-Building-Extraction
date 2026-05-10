# ADCSP-Net

**ADCSP-Net: Adaptive Dual-Context Attention and Spatial-Aware Pooling Network for Remote Sensing Building Extraction**

This repository contains the official PyTorch implementation of ADCSP-Net, as described in our paper published in **[Journal Name]**.

## Overview

ADCSP-Net is a U-Net-based architecture for building extraction from high-resolution remote sensing imagery. It introduces two novel modules:

- **SAPS (Spatial-Aware Pooling Selector)**: An adaptive downsampling module that dynamically fuses average and max pooling with spatially varying weights to preserve structural details.
- **ADCAM (Adaptive Dual-Context Attention Module)**: A dual-attention mechanism integrated into skip connections that jointly models channel and spatial dependencies with high-level semantic guidance.

## Datasets

The code supports the following public datasets:
- WHU Building Dataset
- Massachusetts Building Dataset
- Inria Aerial Image Labeling Dataset

## Requirements

- Python 3.8
- PyTorch 1.10.0
- CUDA 11.1 (recommended)

## Usage

### Training

```bash
python train.py --dataset WHU --epochs 150 --batch_size 4 --lr 1e-4
