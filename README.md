# CenterPoint for Argoverse Data

This repository is a simplified version of the [original CenterPoint repo](https://github.com/tianweiy/CenterPoint), with far less code, designed purely for inference (not for training).

## Installation

Environment:
- Python 3.8.3

**Dependencies**
- SparseConv (build locally)
- DCN (build locally), produce deform_conv_cuda.cpython-38-x86_64-linux-gnu.so
- Iou3dNMS (build locally), produce iou3d_nms_cuda.cpython-38-x86_64-linux-gnu.so
- Pytorch 1.7.1 (check with `python -c "import torch; print(torch.__version__)"`)
- argoverse-api
- CUDA 11.0 (check with `python -c "import torch; print(torch.version.cuda)"`)

If you wish to run `viz_aggregated_sweeps.py`, you must run:
Mayavi Environment: https://github.com/mne-tools/mne-python/blob/master/environment.yml

## Bug Fixed addressed

https://github.com/pytorch/pytorch/issues/29642

use torch::RegisterOperators

Not a problem if you use latest Pytorch?


nvcc fatal   : Unknown option '-Wall'
https://github.com/traveller59/spconv/issues/69
CUDACXX=/usr/local/cuda/bin/nvcc python setup.py bdist_wheel
pip install * --force-reinstall


https://pytorch.org/get-started/previous-versions/


- RuntimeError: /nethome/jlambert30/spconv/src/spconv/indice.cu 274
cuda execution failed with error 98 invalid device function
prepareSubMGridKernel failed
https://github.com/traveller59/spconv/issues/34
Make sure you use the same CUDA version for all installations (set CUDA_HOME before building anything)

## Deformable Convolution 

Added here:
https://github.com/pytorch/vision/pull/1586/files