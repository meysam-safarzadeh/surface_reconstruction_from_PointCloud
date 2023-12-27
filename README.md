# DeepSDF-Based Neural Implicit Surface Reconstruction

## Project Overview

This repository implements a neural implicit surface reconstruction model based on the DeepSDF architecture, as described in the CVPR 2019 paper. It comprises a multi-layer fully-connected neural network that learns to approximate Signed Distance Functions (SDFs) from 3D point samples.

## Repository Structure

- `checkpoints/`: Stores trained model checkpoints.
- `data/`: Dataset directory.
- `model.py`: Neural network architecture.
- `train.py`: Training procedure implementation.
- `utils.py`: Utility functions for training point sampling.

## Requirements

- torch 2.1.2
- numpy 1.26.2
- open3d 0.17.0
- scikit-image 0.20.0

### Installation

First, clone the repository:

```bash
git clone https://github.com/meysam-safarzadeh/surface_reconstruction_from_PointCloud.git
cd surface_reconstruction_from_PointCloud
