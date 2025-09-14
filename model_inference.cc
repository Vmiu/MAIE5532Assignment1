// # Generate C++ model data

// xxd -i mnist_model_quantized.tflite > model_data.cc

// model_inference.cc

#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

// Define constants and global variables
const int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Include your model data
extern const unsigned char mnist_model_quantized_tflite[];
extern const unsigned int mnist_model_quantized_tflite_len;

 

// TODO: Implement setup_model() function
tflite::MicroInterpreter* setup_model() {

    // - Create error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // - Load model from data
    const tflite::Model* model = tflite::GetModel(mnist_model_quantized_tflite);

    // - Create ops resolver and add operations
    static tflite::MicroMutableOpResolver<11> resolver;
    resolver.AddShape();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddRelu();
    resolver.AddStridedSlice();
    resolver.AddPack();

    // - Set up interpreter with memory arena
    static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  
    // Print model size information for debugging
    printf("Model size: %d bytes\n", mnist_model_quantized_tflite_len);
    printf("Arena size: %d bytes\n", kTensorArenaSize);
    printf("Arena address: %p\n", tensor_arena);

    return &static_interpreter;
}
 

// TODO: Implement run_inference() function 
int run_inference(tflite::MicroInterpreter* interpreter, const uint8_t* input_data, size_t input_size) {
    
    // - Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("FAILED: Tensor allocation\n");
        return -1; // Error: allocation failed
    }
    printf("SUCCESS: Tensor allocated\n");
    
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    // Print tensor details for debugging
    printf("Input bytes: %zu (expected: %zu)\n", input->bytes,input_size);
    printf("output bytes: %zu\n", output->bytes);

    printf("Input type=%d scale=%.6f zp=%d bytes=%zu\n",
       input->type, input->params.scale, input->params.zero_point, input->bytes);
    printf("Output type=%d scale=%.6f zp=%d bytes=%zu\n",
       output->type, output->params.scale, output->params.zero_point, output->bytes);
    if (input->bytes != input_size) {
       printf("Input size mismatch: expected %zu, got %d\n", input_size, input->bytes);
        return -1; // Error: size mismatch
    }
    if (output->type != kTfLiteUInt8) {
        printf("Output tensor type mismatch: expected UInt8\n");
        return -1; // Error: type mismatch
    }

    // - Copy input data to input tensor
    memcpy(input->data.uint8, input_data, input_size);

    // - Invoke interpreter
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("FAILED: Invoke\n");
        return -1; // Error: inference failed
    }

    

    // - Return predicted class
    int predicted_class = 0;
    float max_score = 0.0f;
    
    for (int i = 0; i < 10; i++) {
        float score = output->data.f[i];
        if (score > max_score) {
            max_score = score;
            predicted_class = i;
        }
    }
    
    printf("Max score: %f\n", max_score);

    return predicted_class;
}
 

// TODO: Implement main() function

// - Set up model

// - Create test input (28x28 image data)

// - Run inference

// - Print results

 

void create_test_image() {
    // Load 5 sample from MNIST test set

    // Normalize and quantize to uint8
    // Store in test_image array
}
    

int main() {

    // Your implementation here

    // Set up model
    tflite::MicroInterpreter* interpreter = setup_model();
    if (!interpreter) {
        return 1;
    }
    // Create test input (28x28 image data)
    create_test_image(test_image);

    // Run inference
    int predicted_class = run_inference(interpreter, test_image, sizeof(test_image));
    if (predicted_class < 0) {
        printf("Inference failed\n");
        return 1;
    }
    // Print results
    printf("Predicted class: %d\n", predicted_class);

    return 0;

}