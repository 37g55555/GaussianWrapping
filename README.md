> This repository extends [GaussianWrapping](https://github.com/diego1401/GaussianWrapping) with an automated pipeline from object isolation to FBX export.
> For the original GaussianWrapping documentation, please refer to the [original GaussianWrapping README](README_GaussianWrapping.md).

## Installation

### COLMAP

```bash
sudo apt update
sudo apt install -y colmap
```

### GaussianWrapping

```bash
git clone --recurse-submodules https://github.com/diego1401/GaussianWrapping
cd GaussianWrapping
conda create -n gaussian_wrapping python=3.9 -y
conda activate gaussian_wrapping

python -m pip install mapbox-earcut

# Set CUDA paths (adjust version as needed)
CUDA_VERSION=12.1 # or 11.8 (12.1 is best for 4090RTX)

export CPATH=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH

python install.py --cuda_version ${CUDA_VERSION}
```

### Blender

```bash
mkdir -p tools

wget https://download.blender.org/release/Blender4.5/blender-4.5.2-linux-x64.tar.xz \
  -O tools/blender.tar.xz

tar -xJf tools/blender.tar.xz -C tools

ln -sT blender-4.5.2-linux-x64 tools/blender

tools/blender/blender --version

```

## Run

```bash
cd GaussianWrapping
conda activate gaussian_wrapping

export DATASET_PATH=/path/to/dataset
bash run_pipeline.sh
```
