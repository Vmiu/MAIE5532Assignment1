## Project Overview

- **Model**: Quantized MNIST digit classifier (26,888 bytes)
- **Framework**: TensorFlow Lite Micro v2.4.0

## Prerequisites

### System Requirements
- **Operating System**: macOS
- **Compiler**: Clang++
- **Make**: GNU Make 3.82+
- **Python**: 3.13

### Dependencies
- TensorFlow Lite Micro source code (included)
- Standard C++11 libraries
- Math library (`-lm`)

## Project Structure

```
MAIE5532Assignment1/
├── README.md                    # This file
├── Makefile                     # Build configuration
├── model_inference.cc           # Main inference application
├── model_data.cc               # MNIST model data (generated)
├── .gitignore                  # Git ignore patterns
├── requirements.txt            # Python dependencies (if any)
└── tensorflow/                 # TensorFlow Lite Micro source
    └── tensorflow/lite/micro/kernels/
        ├── quantize.cc         # Modified quantize kernel
        └── strided_slice.cc    # Modified strided_slice kernel
```

## Quick Start

### 1. Clone and Navigate
```bash
git clone https://github.com/Vmiu/MAIE5532Assignment1.git
cd MAIE5532Assignment1
```

### 2. Clone Tensorflow v2.4.0 and build
```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.4.0
```
### 3. Move file *quantize.cc* & *strided_slice.cc* to tensorflow/lite/micro/kernels to replace the original files

### 4. Build up the tensorflow library and back to the root dir
``` bash
gmake tensorflow/lite/micro/tools/make/Makefile
cd ..
```

### 4. Setup python environment 
```bash
uv sync
```


## Build System

### Makefile Targets

| Target | Description |
|--------|-------------|
| `gmake` or `make` | Build the project |
| `gmake clean` | Remove build artifacts |
| `gmake run` | Build and run the application |

### Compiler Flags
- `-std=c++11`: C++11 standard
- `-O3`: Maximum optimization
- `-Wall`: All warnings
- `-DTF_LITE_STATIC_MEMORY`: Static memory allocation
- `-DTF_LITE_USE_GLOBAL_*`: Global function usage

## Custom Kernel Modifications

This project includes custom modifications to TensorFlow Lite Micro kernels to support the MNIST model's quantization requirements:

### 1. Quantize Kernel (`tensorflow/tensorflow/lite/micro/kernels/quantize.cc`)
- **Added**: Support for `kTfLiteUInt8` input type
- **Added**: UInt8 to various output type conversions
- **Added**: Int8 to UInt8 conversion support

### 2. StridedSlice Kernel (`tensorflow/tensorflow/lite/micro/kernels/strided_slice.cc`)
- **Added**: Support for `kTfLiteInt32` output type

