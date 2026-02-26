# Research Environment Setup Documentation

**Project**: Small Object Detection on VisDrone  
**Institution**: Research on Edge Devices for Daycare Application  
**Date**: February 7, 2026  
**Hardware**: 2× NVIDIA GeForce RTX 3090 (24GB VRAM each)

---

## Table of Contents
1. [System Information](#system-information)
2. [Core Deep Learning Frameworks](#core-deep-learning-frameworks)
3. [Computer Vision Libraries](#computer-vision-libraries)
4. [Data Processing & Augmentation](#data-processing--augmentation)
5. [Model Architectures & Utilities](#model-architectures--utilities)
6. [Development Tools](#development-tools)
7. [Complete Package List](#complete-package-list)
8. [Installation Commands](#installation-commands)

---

## System Information

### Operating System
- **OS**: Linux (Ubuntu-based)
- **Kernel**: x86_64

### Hardware
- **CPU**: Multi-core processor
- **GPU**: 
  - 2× NVIDIA GeForce RTX 3090 (24GB VRAM each)
  - Total GPU Memory: 48GB
  - NVIDIA Driver: 535.247.01
  - CUDA Version Support: 12.4

### Python Environment
- **Python Version**: 3.11.8
- **Package Manager**: Anaconda (conda) + pip
- **Environment**: base (Anaconda3)
- **Total Packages**: 630+

---

## Core Deep Learning Frameworks

### PyTorch Ecosystem
```bash
torch                    2.5.0          # Latest PyTorch with CUDA 12.4 support
torchvision              0.20.0         # Vision models and transforms
torchmetrics             1.5.1          # Evaluation metrics (mAP, etc.)
pytorch-lightning        2.4.0          # High-level training framework
```

**CUDA Dependencies:**
```bash
nvidia-cuda-cupti-cu12   12.4.127       # CUDA Profiling Tools Interface
nvidia-cuda-nvcc-cu12    12.3.107       # CUDA Compiler
nvidia-cuda-nvrtc-cu12   12.4.127       # CUDA Runtime Compilation
nvidia-cuda-runtime-cu12 12.4.127       # CUDA Runtime
```

### TensorFlow/Keras
```bash
tensorflow               2.12.0         # Deep learning framework
tensorflow-estimator     2.12.0         # High-level TensorFlow API
tensorflow-addons        0.23.0         # Additional TensorFlow operations
tensorflow-io-gcs-filesystem 0.36.0    # GCS support
keras                    2.12.0         # High-level neural networks API
vit-keras                0.1.2          # Vision Transformer for Keras
```

### Other ML Frameworks
```bash
chainer                  7.8.1          # Alternative deep learning framework
scikit-learn             1.5.2          # Traditional ML algorithms
catboost                 1.2.7          # Gradient boosting library
xgboost                  (installed)    # Gradient boosting
lightgbm                 (installed)    # Light gradient boosting
```

---

## Computer Vision Libraries

### Core CV Libraries
```bash
opencv-python            4.9.0.80       # OpenCV with GUI
opencv-python-headless   4.11.0.86      # OpenCV without GUI (for servers)
scikit-image             0.22.0         # Image processing algorithms
pillow                   10.4.0         # Python Imaging Library
imageio                  (installed)    # Reading/writing images
```

### Object Detection & Segmentation
```bash
# Torchvision Detection Models
- Faster R-CNN (ResNet50-FPN)
- Mask R-CNN
- RetinaNet
- SSD
- FCOS

# Deformable Convolutions
torchvision.ops.DeformConv2d (built-in)
```

### Data Augmentation
```bash
albumentations           2.0.8          # Advanced augmentation library
albucore                 0.0.24         # Albumentations core
imgaug                   (installed)    # Image augmentation
```

### Face Recognition (Additional Projects)
```bash
# Used in "Child Identity Verification" project
facenet-pytorch          (installable)  # Face recognition embeddings
mtcnn                    (installable)  # Face detection
```

### Pose Estimation (Additional Projects)
```bash
mediapipe                (installable)  # Google's ML solutions (pose, face, hands)
# Used in "Action Recognition" project
```

---

## Data Processing & Augmentation

### Numerical Computing
```bash
numpy                    1.26.4         # Array operations
scipy                    1.11.4         # Scientific computing
numba                    0.58.1         # JIT compiler for Python
```

### Data Analysis
```bash
pandas                   2.1.4          # Data manipulation
pyarrow                  (installed)    # Apache Arrow for data
tables                   3.9.2          # HDF5 file support
h5py                     3.9.0          # HDF5 Python interface
```

### Visualization
```bash
matplotlib               3.8.0          # Plotting library
matplotlib-base          3.8.0
matplotlib-inline        0.1.6
seaborn                  0.13.2         # Statistical visualization
plotly                   5.9.0          # Interactive plots
bokeh                    3.3.4          # Interactive visualization
```

### Image Processing
```bash
imagecodecs              2024.6.1       # Image codecs (JPEG, PNG, TIFF, etc.)
imageio                  (installed)    # Image I/O
pillow                   10.4.0         # PIL fork
scikit-image             0.22.0         # Image algorithms
```

---

## Model Architectures & Utilities

### Transformers & NLP (for multimodal)
```bash
transformers             4.42.4         # Hugging Face transformers
huggingface-hub          0.24.0         # Model hub access
tokenizers               0.19.1         # Fast tokenization
safetensors              (installed)    # Safe tensor storage
```

### Model Utilities
```bash
timm                     (installable)  # PyTorch Image Models
efficientnet-pytorch     (installable)  # EfficientNet implementation
pretrainedmodels         (installable)  # Pre-trained models
```

### Acceleration & Optimization
```bash
accelerate               1.0.1          # Distributed training
bitsandbytes             0.44.1         # 8-bit optimizers
onnx                     (installed)    # ONNX format support
onnxruntime              (installed)    # ONNX runtime
openvino                 (installable)  # Intel optimization
```

---

## Development Tools

### Jupyter & Notebooks
```bash
jupyter                  1.0.0          # Jupyter metapackage
jupyter-client           8.6.0
jupyter-console          6.6.3
jupyter-core             5.5.0
jupyter-server           2.10.0
jupyterlab               4.0.11
jupyterlab-server        2.25.1
notebook                 7.0.8
ipython                  8.20.0         # Interactive Python
ipykernel                6.28.0         # IPython kernel
ipywidgets               8.1.2          # Interactive widgets
```

### Code Quality & Formatting
```bash
black                    23.11.0        # Code formatter
autopep8                 2.0.4          # PEP8 formatter
flake8                   6.1.0          # Linter
pylint                   2.16.2         # Code analysis
mypy                     1.8.0          # Static type checker
isort                    5.13.2         # Import sorter
```

### Documentation
```bash
sphinx                   7.2.6          # Documentation generator
numpydoc                 1.5.0          # NumPy style docstrings
```

### Version Control & Collaboration
```bash
git                      (system)       # Version control
gitpython                3.1.37         # Git Python interface
```

### Testing
```bash
pytest                   7.4.0          # Testing framework
pytest-cov               4.0.0          # Coverage plugin
unittest                 (built-in)     # Python unit testing
```

---

## Complete Package List

### A-C
```
absl-py                  2.1.0
accelerate               1.0.1
aiobotocore              2.7.0
aiohttp                  3.9.3
albumentations           2.0.8
albucore                 0.0.24
altair                   5.0.1
anaconda                 2024.02-1
annotated-types          0.6.0
antlr4-python3-runtime   4.9.3
anyio                    4.2.0
appdirs                  1.4.4
archspec                 0.2.3
argon2-cffi              21.3.0
arrow                    1.2.3
astor                    0.8.1
astroid                  2.14.2
astropy                  5.3.4
asttokens                2.0.5
astunparse               1.6.3
async-lru                2.0.4
atomicwrites             1.4.0
attrs                    23.1.0
autopep8                 2.0.4
babel                    2.11.0
beautifulsoup4           4.12.2
binaryornot              0.4.4
bitsandbytes             0.44.1
black                    23.11.0
bleach                   4.1.0
blinker                  1.6.2
bokeh                    3.3.4
boltons                  23.0.0
boto3                    1.28.64
botocore                 1.31.64
bottleneck               1.3.7
brotli                   1.1.0
bzip2                    1.0.8
cachetools               4.2.2
catboost                 1.2.7
certifi                  2024.8.30
cffi                     1.16.0
chainer                  7.8.1
chardet                  4.0.0
click                    8.1.7
cloudpickle              2.2.1
colorama                 0.4.6
```

### D-H
```
dask                     2023.11.0
datasets                 2.14.6
debugpy                  1.6.7
decorator                5.1.1
defusedxml               0.7.1
dill                     0.3.7
distributed              2023.11.0
docutils                 0.18.1
entrypoints              0.4
et-xmlfile               1.1.0
exceptiongroup           1.2.0
fastjsonschema           2.16.2
filelock                 3.13.1
flake8                   6.1.0
fonttools                4.51.0
frozenlist               1.4.0
fsspec                   2023.10.0
gast                     0.4.0
gensim                   4.3.0
gitdb                    4.0.7
gitpython                3.1.37
google-auth              2.23.3
google-auth-oauthlib     1.0.0
google-pasta             0.2.0
graphviz                 0.20.1
greenlet                 3.0.1
grpcio                   1.60.0
h5py                     3.9.0
heapdict                 1.0.1
holoviews                1.18.3
huggingface-hub          0.24.0
hvplot                   0.9.2
hyperopt                 0.2.7
```

### I-N
```
idna                     3.4
imagecodecs              2024.6.1
imageio                  2.33.1
imagesize                1.4.1
importlib-metadata       7.0.1
importlib-resources      6.1.1
iniconfig                2.0.0
intake                   0.7.0
intervaltree             3.1.0
ipykernel                6.28.0
ipython                  8.20.0
ipywidgets               8.1.2
isort                    5.13.2
itsdangerous             2.1.2
jedi                     0.18.1
jinja2                   3.1.3
joblib                   1.4.2
json5                    0.9.6
jsonpatch                1.32
jsonpointer              2.1
jsonschema               4.19.2
jupyter                  1.0.0
jupyter-client           8.6.0
jupyter-console          6.6.3
jupyter-core             5.5.0
jupyter-events           0.8.0
jupyter-lsp              2.2.0
jupyter-server           2.10.0
jupyterlab               4.0.11
jupyterlab-pygments      0.1.2
jupyterlab-server        2.25.1
jupyterlab-widgets       3.0.9
keras                    2.12.0
keyring                  23.13.1
kiwisolver               1.4.4
lazy-object-proxy        1.10.0
libclang                 16.0.6
lightning                2.4.0
lightning-utilities      0.11.10
llvmlite                 0.41.0
locket                   1.0.0
lxml                     4.9.3
markdown                 3.4.1
markupsafe               2.1.3
matplotlib               3.8.0
matplotlib-base          3.8.0
matplotlib-inline        0.1.6
mccabe                   0.7.0
mistune                  2.0.4
mpmath                   1.3.0
msgpack                  1.0.3
multidict                6.0.4
multipledispatch         0.6.0
mypy                     1.8.0
nbclient                 0.8.0
nbconvert                7.10.0
nbformat                 5.9.2
nest-asyncio             1.6.0
networkx                 3.1
nltk                     3.8.1
notebook                 7.0.8
numba                    0.58.1
numexpr                  2.8.7
numpy                    1.26.4
numpydoc                 1.5.0
```

### O-S
```
oauthlib                 3.2.2
opencv-python            4.9.0.80
opencv-python-headless   4.11.0.86
openpyxl                 3.1.2
opt-einsum               3.3.0
optuna                   3.5.0
packaging                23.1
pandas                   2.1.4
pandocfilters            1.5.0
panel                    1.3.8
param                    2.0.2
parso                    0.8.3
partd                    1.4.1
patchelf                 0.17.2.1
pathspec                 0.10.3
patsy                    0.5.3
pexpect                  4.8.0
pickleshare              0.7.5
pillow                   10.4.0
pip                      23.3.1
pkginfo                  1.9.6
platformdirs             3.10.0
plotly                   5.9.0
pluggy                   1.0.0
ply                      3.11
pooch                    1.8.0
prometheus-client        0.14.1
prompt-toolkit           3.0.43
protobuf                 4.23.4
psutil                   5.9.0
ptyprocess               0.7.0
pure-eval                0.2.2
py                       1.11.0
py-cpuinfo               9.0.0
pyarrow                  11.0.0
pyasn1                   0.4.8
pyasn1-modules           0.2.8
pycparser                2.21
pycodestyle              2.11.1
pycurl                   7.45.2
pydantic                 2.5.0
pydantic-core            2.14.5
pydocstyle               6.3.0
pyerfa                   2.0.0
pyflakes                 3.1.0
pygments                 2.15.1
pyjwt                    2.4.0
pylint                   2.16.2
pyopenssl                23.2.0
pyparsing                3.0.9
pyqt5                    5.15.10
pyqt5-sip                12.13.0
pysocks                  1.7.1
pytest                   7.4.0
pytest-cov               4.0.0
python-dateutil          2.8.2
python-json-logger       2.0.7
python-lsp-jsonrpc       1.0.0
python-lsp-server        1.7.2
python-slugify           5.0.2
pytorch-lightning        2.4.0
pytz                     2023.3.post1
pyviz-comms              2.3.0
pywavelets               1.5.0
pyyaml                   6.0.1
pyzmq                    25.1.2
qdarkstyle               3.0.2
qstylizer                0.2.2
qtawesome                1.2.2
qtconsole                5.4.2
qtpy                     2.4.1
regex                    2023.10.3
requests                 2.31.0
requests-oauthlib        1.3.1
responses                0.13.3
rfc3339-validator        0.1.4
rfc3986-validator        0.1.1
rope                     1.10.0
rsa                      4.9
rtree                    1.0.1
ruamel.yaml              0.17.21
safetensors              0.4.2
scikit-image             0.22.0
scikit-learn             1.5.2
scipy                    1.11.4
seaborn                  0.13.2
send2trash               1.8.2
setuptools               68.2.2
six                      1.16.0
smart-open               5.2.1
sniffio                  1.3.0
snowballstemmer          2.2.0
sortedcontainers         2.4.0
soupsieve                2.5
sphinx                   7.2.6
sphinxcontrib-applehelp  1.0.2
sphinxcontrib-devhelp    1.0.2
sphinxcontrib-htmlhelp   2.0.0
sphinxcontrib-jsmath     1.0.1
sphinxcontrib-qthelp     1.0.3
sphinxcontrib-serializinghtml 1.1.5
spyder                   5.4.3
spyder-kernels           2.4.4
sqlalchemy               2.0.25
stack-data               0.2.0
statsmodels              0.14.0
sympy                    1.12
```

### T-Z
```
tables                   3.9.2
tabulate                 0.9.0
tblib                    1.7.0
tenacity                 8.2.2
tensorboard              2.12.3
tensorboard-data-server  0.7.0
tensorflow               2.12.0
tensorflow-addons        0.23.0
tensorflow-estimator     2.12.0
tensorflow-io-gcs-filesystem 0.36.0
termcolor                2.1.0
terminado                0.17.1
text-unidecode           1.3
textdistance             4.2.1
threadpoolctl            3.5.0
three-merge              0.1.1
tifffile                 2023.4.12
tinycss2                 1.2.1
tokenizers               0.19.1
toml                     0.10.2
tomli                    2.0.1
tomlkit                  0.11.1
toolz                    0.12.0
torch                    2.5.0
torchmetrics             1.5.1
torchvision              0.20.0
tornado                  6.3.3
tqdm                     4.65.0
traitlets                5.7.1
transformers             4.42.4
truststore               0.8.0
typeguard                2.13.3
typing-extensions        4.9.0
tzdata                   2023.3
ujson                    5.4.0
unicodedata2             15.1.0
unidecode                1.2.0
urllib3                  2.0.7
vit-keras                0.1.2
wcwidth                  0.2.5
webencodings             0.5.1
websocket-client         0.58.0
werkzeug                 2.2.3
whatthepatch             1.0.2
wheel                    0.41.2
widgetsnbextension       4.0.5
wrapt                    1.14.1
wurlitzer                3.0.2
xarray                   2023.6.0
xlrd                     2.0.1
xlsxwriter               3.1.1
xxhash                   3.1.0
xz                       5.4.5
yaml                     0.2.5
yapf                     0.31.0
yarl                     1.9.3
zeromq                   4.3.5
zict                     3.0.0
zipp                     3.17.0
zlib                     1.2.13
zope.interface           5.4.0
zstandard                0.19.0
zstd                     1.5.5
```

---

## Installation Commands

### 1. Setup Base Anaconda Environment
```bash
# Download and install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# Create environment (or use base)
conda create -n research python=3.11
conda activate research
```

### 2. Install PyTorch with CUDA Support
```bash
# PyTorch 2.5.0 with CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# Or latest stable
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Computer Vision Libraries
```bash
# Core CV libraries
pip install opencv-python opencv-python-headless
pip install scikit-image
pip install albumentations
conda install -c conda-forge pillow

# Object detection utilities
pip install torchmetrics
pip install pytorch-lightning
```

### 4. Install TensorFlow (Optional)
```bash
pip install tensorflow==2.12.0
pip install tensorflow-addons
pip install keras
```

### 5. Install Data Science Stack
```bash
# Core scientific computing
conda install numpy pandas scipy matplotlib seaborn

# Jupyter
conda install jupyter jupyterlab notebook ipython ipywidgets

# Additional tools
pip install scikit-learn
pip install xgboost lightgbm catboost
```

### 6. Install Development Tools
```bash
# Code quality
pip install black autopep8 flake8 pylint mypy isort

# Testing
pip install pytest pytest-cov

# Documentation
pip install sphinx numpydoc
```

### 7. Install Transformers & NLP (Optional)
```bash
pip install transformers
pip install huggingface-hub tokenizers
pip install accelerate bitsandbytes
```

### 8. Verify Installation
```bash
# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Check TorchVision
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

# Check GPU
nvidia-smi

# Check TensorFlow (if installed)
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

---

## Project-Specific Dependencies

### Small Object Detection (VisDrone)
```python
# Already installed
torch>=2.5.0
torchvision>=0.20.0
torchmetrics>=1.5.1
albumentations>=2.0.8
opencv-python>=4.9.0
numpy>=1.26.4
pandas>=2.1.4
matplotlib>=3.8.0
tqdm>=4.65.0
pillow>=10.4.0
```

### Installation for This Research
```bash
cd /home/mahin/Documents/notebook/small-object-detection
pip install -r requirements.txt  # If exists

# Or install manually
pip install torch torchvision torchmetrics albumentations opencv-python numpy pandas matplotlib tqdm pillow
```

---

## Hardware Specifications Summary

```
GPU: 2× NVIDIA GeForce RTX 3090
├─ VRAM: 24GB per GPU (48GB total)
├─ CUDA Compute Capability: 8.6
├─ Driver Version: 535.247.01
├─ CUDA Runtime: 12.4
└─ Memory Bandwidth: 936 GB/s per GPU

CPU: Multi-core x86_64 processor
RAM: Sufficient for large-scale training
Storage: SSD (recommended for fast data loading)
OS: Linux (Ubuntu-based distribution)
```

---

## Training Configuration Used

### Baseline Training (38.02% mAP)
```python
batch_size = 4
learning_rate = 0.005
optimizer = SGD
epochs = 50
backbone = ResNet50-FPN (pretrained)
```

### SimplifiedDPA Training (43.44% mAP) ✅ BEST
```python
batch_size = 4
learning_rate = 0.005
optimizer = SGD
epochs = ~20
backbone = ResNet50-FPN (pretrained)
enhancement = SimplifiedDPA (P2, P3, P4)
```

### CD-DPA Training (35.29% mAP) ❌ FAILED
```python
batch_size = 4 (effective 16 with accumulation)
learning_rate = 1e-4
optimizer = AdamW
epochs = 7 (early stopped)
backbone = ResNet50-FPN (pretrained)
enhancement = CD-DPA (P2, P3, P4)
mixed_precision = True (FP16)
gradient_checkpointing = True
gradient_accumulation_steps = 4
```

---

## Performance Summary

| Model | mAP@0.5 | Training Time | Parameters | Status |
|-------|---------|---------------|------------|--------|
| Baseline Faster R-CNN | 38.02% | ~3-4 hours | 41.3M | ✓ Reference |
| SimplifiedDPA | 43.44% | ~2-3 hours | 44.5M | ✅ **BEST** |
| CD-DPA | 35.29% | ~2.5 hours | 48.2M | ❌ Failed |

---

## Notes

1. **SimplifiedDPA is the winning architecture** - Simple dual-path attention works better than complex deformable cascade
2. **GPU Memory** - 24GB is sufficient for all experiments with proper memory optimization
3. **CUDA Version** - Using CUDA 12.4 with PyTorch 2.5.0 (latest stable)
4. **Mixed Precision** - FP16 training reduces memory by ~40% but didn't help CD-DPA performance
5. **Python 3.11** - Latest stable Python version with best performance

---

## Export Package Lists

Your complete package lists are saved at:
- **Conda packages**: `/tmp/conda_packages.txt`
- **Pip packages**: `/tmp/pip_packages.txt`

To export for sharing:
```bash
# Conda environment
conda env export > environment.yml

# Pip requirements
pip freeze > requirements.txt

# Copy to project
cp /tmp/conda_packages.txt ~/Documents/notebook/small-object-detection/conda_packages.txt
cp /tmp/pip_packages.txt ~/Documents/notebook/small-object-detection/pip_packages.txt
```

To recreate environment:
```bash
# From conda
conda env create -f environment.yml

# From pip
pip install -r requirements.txt
```

---

**Last Updated**: February 7, 2026  
**Total Packages**: 630+  
**Python Version**: 3.11.8  
**PyTorch Version**: 2.5.0  
**CUDA Version**: 12.4
