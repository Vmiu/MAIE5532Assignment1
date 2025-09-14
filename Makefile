# Standalone Makefile for model_inference example

# Check make version (requires gmake 3.82+)
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
  $(error "Requires gmake version 3.82 or later (current is $(MAKE_VERSION)). Please use 'gmake' instead of 'make'")
endif

# Compiler settings
CXX := clang++
CXXFLAGS := -std=c++11 -O3 -Wall \
           -DTF_LITE_STATIC_MEMORY \
           -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS \
           -DTF_LITE_USE_GLOBAL_MIN \
           -DTF_LITE_USE_GLOBAL_MAX
LDFLAGS := -lm

# Include paths
INCLUDES := \
    -I. \
    -I../tensorflow \
    -Itensorflow \
    -I../tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
    -I../tensorflow/lite/micro/tools/make/downloads/gemmlowp \
    -I../tensorflow/lite/micro/tools/make/downloads/ruy \
    -I../tensorflow/lite/micro/tools/make/downloads \
    -Itensorflow/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
    -Itensorflow/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
    -Itensorflow/tensorflow/lite/micro/tools/make/downloads/ruy

# Application source files
APP_SRCS := \
    model_inference.cc \
    model_data.cc

# Core TFLM source files
CORE_SRCS := \
    tensorflow/tensorflow/lite/micro/micro_error_reporter.cc \
    tensorflow/tensorflow/lite/micro/micro_interpreter.cc \
    tensorflow/tensorflow/lite/micro/micro_allocator.cc \
    tensorflow/tensorflow/lite/micro/simple_memory_allocator.cc \
    tensorflow/tensorflow/lite/micro/memory_helpers.cc \
    tensorflow/tensorflow/lite/micro/debug_log.cc \
    tensorflow/tensorflow/lite/micro/micro_string.cc \
    tensorflow/tensorflow/lite/micro/micro_time.cc \
    tensorflow/tensorflow/lite/micro/micro_utils.cc \
    tensorflow/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
    tensorflow/tensorflow/lite/c/common.c \
    tensorflow/tensorflow/lite/core/api/error_reporter.cc \
    tensorflow/tensorflow/lite/core/api/flatbuffer_conversions.cc \
    tensorflow/tensorflow/lite/core/api/op_resolver.cc \
    tensorflow/tensorflow/lite/kernels/internal/quantization_util.cc \
    tensorflow/tensorflow/lite/kernels/kernel_util.cc \
    tensorflow/tensorflow/lite/schema/schema_utils.cc

# Kernel source files (exactly matching compile_part3.sh)
KERNEL_SRCS := \
    tensorflow/tensorflow/lite/micro/kernels/conv.cc \
    tensorflow/tensorflow/lite/micro/kernels/pooling.cc \
    tensorflow/tensorflow/lite/micro/kernels/fully_connected.cc \
    tensorflow/tensorflow/lite/micro/kernels/softmax.cc \
    tensorflow/tensorflow/lite/micro/kernels/reshape.cc \
    tensorflow/tensorflow/lite/micro/kernels/quantize.cc \
    tensorflow/tensorflow/lite/micro/kernels/dequantize.cc \
    tensorflow/tensorflow/lite/micro/kernels/activations.cc \
    tensorflow/tensorflow/lite/micro/kernels/kernel_util.cc \
    tensorflow/tensorflow/lite/micro/kernels/shape.cc \
    tensorflow/tensorflow/lite/micro/kernels/strided_slice.cc \
    tensorflow/tensorflow/lite/micro/kernels/pack.cc

# Combine all source files
ALL_SRCS := $(APP_SRCS) $(CORE_SRCS) $(KERNEL_SRCS)

# Target executable
TARGET := model_inference

# Build target
$(TARGET): check-dependencies $(APP_SRCS)

# Build target
$(TARGET): check-dependencies $(OBJ_FILES)
	@echo "🔄 Building TensorFlow Lite Micro application..."
	@echo "📝 Using TensorFlow Lite Micro v2.4.0 compatible build"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(ALL_SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "✅ Build complete: $(TARGET)"
	@echo "📊 Binary size: $$(ls -lh $(TARGET) | awk '{print $$5}')"

# Run the application
run: $(TARGET)
	@echo "🚀 Running TensorFlow Lite Micro inference..."
	./$(TARGET)

# Clean generated files
clean:
	rm -f $(TARGET)
	@echo "Clean complete"

.PHONY: all run clean check-dependencies info help
