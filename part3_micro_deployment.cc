// # Generate C++ model data

// xxd -i mnist_model_quantized.tflite > model_data.cc

// model_inference.cc

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Define constants and global variables
const int kTensorArenaSize = 40 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Include your model data
extern const unsigned char mnist_model_quantized_tflite[];
extern const unsigned int mnist_model_quantized_tflite_len;

// Memory usage analysis structure
struct MemoryAnalysis {
    size_t flash_usage;      // Model size in flash memory
    size_t ram_usage;        // Runtime RAM usage
    size_t arena_usage;      // Tensor arena usage
    size_t total_footprint;  // Total memory footprint
};

// Function to analyze memory usage
MemoryAnalysis analyze_memory_usage(tflite::MicroInterpreter* interpreter) {
    MemoryAnalysis analysis;
    
    // Flash usage (model size)
    analysis.flash_usage = mnist_model_quantized_tflite_len;
    
    // RAM usage (tensor arena)
    analysis.arena_usage = kTensorArenaSize;
    
    // Total footprint
    analysis.total_footprint = analysis.flash_usage + analysis.arena_usage;
    
    return analysis;
}

// Function to print memory analysis report
void print_memory_analysis(const MemoryAnalysis& analysis) {
    printf("\n");
    printf("========================================\n");
    printf("        MEMORY USAGE ANALYSIS\n");
    printf("========================================\n");
    printf("FLASH MEMORY (Persistent Storage):\n");
    printf("  Model size:        %zu bytes (%.2f KB)\n", 
           analysis.flash_usage, analysis.flash_usage / 1024.0f);
    printf("\n");
    printf("RAM MEMORY (Runtime Memory):\n");
    printf("  Tensor arena:      %zu bytes (%.2f KB)\n", 
           analysis.arena_usage, analysis.arena_usage / 1024.0f);
    printf("\n");
    printf("TOTAL MEMORY FOOTPRINT:\n");
    printf("  Flash + RAM:       %zu bytes (%.2f KB)\n", 
           analysis.total_footprint, analysis.total_footprint / 1024.0f);
    printf("  Total footprint:   %.2f MB\n", analysis.total_footprint / (1024.0f * 1024.0f));
    printf("========================================\n");
}

 

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
    printf("\n");
    printf("========================================\n");
    printf("        MODEL CONFIGURATION\n");
    printf("========================================\n");
    printf("Model size:        %d bytes (%.2f KB)\n", 
           mnist_model_quantized_tflite_len, mnist_model_quantized_tflite_len / 1024.0f);
    printf("Arena size:        %d bytes (%.2f KB)\n", 
           kTensorArenaSize, kTensorArenaSize / 1024.0f);
    printf("Arena address:     %p\n", tensor_arena);
    printf("========================================\n");

    return &static_interpreter;
}
 

// TODO: Implement run_inference() function 
int run_inference(tflite::MicroInterpreter* interpreter, const uint8_t* input_data, size_t input_size) {
    
    // - Allocate tensors
    printf("\n");
    printf("========================================\n");
    printf("        TENSOR ALLOCATION\n");
    printf("========================================\n");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("ERROR: Tensor allocation failed\n");
        return -1; // Error: allocation failed
    }
    printf("SUCCESS: Tensors allocated successfully\n");
    printf("========================================\n");
    
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    // Print tensor details for debugging
    printf("\n");
    printf("========================================\n");
    printf("        TENSOR INFORMATION\n");
    printf("========================================\n");
    printf("Input tensor:\n");
    printf("  Size:             %zu bytes (expected: %zu)\n", input->bytes, input_size);
    printf("  Type:             %d\n", input->type);
    printf("  Scale:            %.6f\n", input->params.scale);
    printf("  Zero point:       %d\n", input->params.zero_point);
    printf("\n");
    printf("Output tensor:\n");
    printf("  Size:             %zu bytes\n", output->bytes);
    printf("  Type:             %d\n", output->type);
    printf("  Scale:            %.6f\n", output->params.scale);
    printf("  Zero point:       %d\n", output->params.zero_point);
    printf("========================================\n");

    // - Copy input data to input tensor
    printf("\n");
    printf("========================================\n");
    printf("        INFERENCE EXECUTION\n");
    printf("========================================\n");
    printf("Copying input data to tensor...\n");
    memcpy(input->data.uint8, input_data, input_size);
    printf("Input data copied successfully\n");

    // - Invoke interpreter
    printf("Invoking interpreter...\n");
    if (interpreter->Invoke() != kTfLiteOk) {
        printf("ERROR: Inference execution failed\n");
        return -1; // Error: inference failed
    }
    printf("Inference completed successfully\n");
    printf("========================================\n");

    

    // - Return predicted class
    int predicted_class = 0;
    float max_score = 0.0f;
    
    printf("\n");
    printf("========================================\n");
    printf("        PREDICTION RESULTS\n");
    printf("========================================\n");
    printf("Processing output probabilities...\n");
    for (int i = 0; i < 10; i++) {
        // Convert quantized uint8 to float using scale and zero point
        float score = (output->data.uint8[i] - output->params.zero_point) * output->params.scale;
        printf("Class %d: raw=%d, dequantized=%.6f\n", i, output->data.uint8[i], score);
        if (score > max_score) {
            max_score = score;
            predicted_class = i;
        }
    }
    
    printf("Predicted class:    %d\n", predicted_class);
    printf("Confidence score:   %.6f\n", max_score);
    printf("========================================\n");

    return predicted_class;
}
 

// TODO: Implement main() function

// - Set up model

// - Create test input (28x28 image data)

// - Run inference

// - Print results

 

void create_test_image(uint8_t* test_image) {
    printf("\n");
    printf("========================================\n");
    printf("        TEST IMAGE GENERATION\n");
    printf("========================================\n");
    printf("Creating 28x28 test image...\n");
    
    // Initialize image with zeros
    memset(test_image, 0, 28 * 28);

    // Draw a vertical line in the center
    for (int i = 10; i < 18; i++) {
        test_image[i * 28 + 14] = 255;
    }
    
    printf("Test image created successfully\n");
    printf("Image size:         %d bytes\n", 28 * 28);
    printf("Pattern:            Vertical line in center\n");
    printf("========================================\n");
}
    

int main() {
    printf("\n");
    printf("========================================\n");
    printf("   TENSORFLOW LITE MICRO INFERENCE\n");
    printf("========================================\n");
    printf("Starting MNIST digit classification...\n");
    printf("========================================\n");

    // Set up model
    tflite::MicroInterpreter* interpreter = setup_model();
    if (!interpreter) {
        printf("\nERROR: Failed to setup model\n");
        return 1;
    }
    
    // Create test input (28x28 image data)
    uint8_t test_image[28 * 28];
    create_test_image(test_image);

    // Run inference
    int predicted_class = run_inference(interpreter, test_image, sizeof(test_image));
    if (predicted_class < 0) {
        printf("\nERROR: Inference failed\n");
        return 1;
    }
    
    // Perform memory analysis
    MemoryAnalysis memory = analyze_memory_usage(interpreter);
    print_memory_analysis(memory);
    
    // Print final results
    printf("\n");
    printf("========================================\n");
    printf("        FINAL RESULTS\n");
    printf("========================================\n");
    printf("Predicted digit:    %d\n", predicted_class);
    printf("Status:             SUCCESS\n");
    printf("========================================\n");
    printf("\nInference completed successfully!\n");

    return 0;
}