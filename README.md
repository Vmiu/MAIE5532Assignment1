## Project Overview

- **Model**: TFLM Model(26,888 bytes)
- **Framework**: TensorFlow Lite Micro v2.4.0

## Prerequisites

### System Requirements
- **Operating System**: macOS
- **Compiler**: Clang++
- **Make**: GNU Make 3.82+
- **Python**: 3.13


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

## Installation Guide for macOS

### 1. **System Requirements Check**
First, verify your system meets the requirements:
- macOS (you're already on this)
- Python 3.13
- Clang++ compiler (comes with Xcode Command Line Tools)
- GNU Make 3.82+

### 2. **Install Xcode Command Line Tools**
```bash
xcode-select --install
```

### 3. **Install Python 3.13**
Since the project requires Python 3.13, install it using Homebrew:
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.13
brew install python@3.13
```

### 4. **Install UV (Python Package Manager)(*optional*)**
The project uses `uv` for dependency management:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to your shell profile
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 5. **Install GNU Make 3.82+**
macOS comes with an older version of make, so install the newer version:
```bash
brew install make
```

### 6. **Set up the Project Environment**
Navigate to your project directory and set up the Python environment:
```bash
git clone https://github.com/Vmiu/MAIE5532Assignment1.git
cd /Users/k/Documents/MV/MAIE5532Assignment1

# Install Python dependencies using uv
uv sync
```

### 7. **Set up TensorFlow Lite Micro**
The project requires TensorFlow v2.4.0 source code. Follow these steps:
```bash
# Clone TensorFlow v2.4.0 (if not already done)
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.4.0

# Copy the custom kernel files
cp ../quantize.cc tensorflow/lite/micro/kernels/
cp ../strided_slice.cc tensorflow/lite/micro/kernels/

# Build TensorFlow Lite Micro
gmake -f tensorflow/lite/micro/tools/make/Makefile

# Return to project root
cd ..
```

### 8. **Verify Installation**
Test that everything is working:
```bash
# Check Python version
python3.13 --version

# Check make version
gmake --version

# Check clang++ version
clang++ --version

# Test Python dependencies
python3.13 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### 9. **Build and Run the Project**
```bash
# Build the C++ application
gmake

# Run the application
gmake run

# Or run Python scripts
python3.13 part1_tensorflow.py
python3.13 part2_tflite_conversion.py
```

## Quick Start

After completing the installation above, you can quickly run the project:

```bash
# Build and run the C++ inference application
gmake run

# Or run individual Python components
python3.13 part1_tensorflow.py      # Train the model
python3.13 part2_tflite_conversion.py  # Convert to TFLite
```

## Custom Kernel Modifications

### 1. Quantize Kernel (`tensorflow/tensorflow/lite/micro/kernels/quantize.cc`)
- **Added**: Support for `kTfLiteUInt8` input type
- **Added**: UInt8 to various output type conversions
- **Added**: Int8 to UInt8 conversion support

### 2. StridedSlice Kernel (`tensorflow/tensorflow/lite/micro/kernels/strided_slice.cc`)
- **Added**: Support for `kTfLiteInt32` output type

