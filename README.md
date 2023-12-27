# DeepSDF-Based Neural Implicit Surface Reconstruction

## Project Overview

This repository contains the implementation of a neural implicit surface reconstruction model based on the DeepSDF architecture (CVPR 2019[^1]). The model, model.py, is a multi-layer fully-connected neural network that learns to approximate Signed Distance Functions (SDFs) from 3D point samples and their SDF values. The architecture involves 8 fully-connected layers with a unique concatenation mechanism after the fourth layer. The training procedure (train.py) minimizes the clamped L1 loss between predicted and actual SDF values of sample points. The utils.py script is responsible for sampling training points and their SDF values.



[^1]: Park, Jeong Joon, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. "Deepsdf: Learning continuous signed distance functions for shape representation." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 165-174. 2019.



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
pip install -r requirements.txt
```

## References

- Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S. (2019). Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 165-174).