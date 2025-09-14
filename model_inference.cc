Create a C++ implementation that uses the converted model:

// model_inference.cc

#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

 

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
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return nullptr;
    }

    // - Create ops resolver
    static tflite::AllOpsResolver resolver;

    // - Set up interpreter with memory arena
    static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  
    if (static_interpreter.AllocateTensors() != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
        return nullptr;
    }

    return &static_interpreter;
}
 

// TODO: Implement run_inference() function 
int run_inference(tflite::MicroInterpreter* interpreter, const uint8_t* input_data, size_t input_size) {

    // - Copy input data to model input tensor
    TfLiteTensor* input = interpreter->input(0);
    if (input->bytes != input_size) {
        return -1; // Error: input size mismatch
    }
    memcpy(input->data.uint8, input_data, input_size);

    // - Invoke interpreter
    if (interpreter->Invoke() != kTfLiteOk) {
        return -1; // Error: inference failed
    }

    // - Read output from output tensor
    TfLiteTensor* output = interpreter->output(0);

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

    return predicted_class;
}
 

// TODO: Implement main() function

// - Set up model

// - Create test input (28x28 image data)

// - Run inference

// - Print results

 

const int kTensorArenaSize = 60 * 1024;

uint8_t tensor_arena[kTensorArenaSize];

void create_test_image(uint8_t* image) {
  // Initialize to all zeros (black)
  memset(image, 0, 28 * 28);
  
  // Draw a simple digit (e.g., number 7)
  for (int i = 5; i < 20; i++) {
    image[i + 28 * 5] = 255;  // Horizontal line at the top
  }
  for (int j = 6; j < 23; j++) {
    image[18 + 28 * j] = 255; // Vertical line on the right
  }
}

int main() {

    // Your implementation here

    // Set up model
    tflite::MicroInterpreter* interpreter = setup_model();
    if (!interpreter) {
        return 1;
    }
    // Create test input (28x28 image data)
    uint8_t test_image[28 * 28];
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