import tensorflow as tf
from part1 import load_and_preprocess_data
import os
import numpy as np

def convert_to_tflite(model_path, quantize=False):

    """

    Convert TensorFlow model to TensorFlow Lite format.

   

    Args:

        model_path (str): Path to saved TensorFlow model

        quantize (bool): Whether to apply quantization

       

    Returns:

        bytes: TensorFlow Lite model data

    """

    # TODO: Load the saved model
    model = tf.keras.models.load_model(model_path)

    # TODO: Create TensorFlow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # TODO: Apply quantization if requested
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Create a representative dataset from training data
        _, _, x_test, _ = load_and_preprocess_data()
        def representative_dataset():
            for data in tf.data.Dataset.from_tensor_slices(x_test).batch(1).take(100):
                yield [tf.dtypes.cast(data, tf.float32)]
        converter.representative_dataset = representative_dataset
        
    # TODO: Convert model and return tflite data
    tflite_model = converter.convert()
    
    return tflite_model

 

def analyze_model_size(tf_model_path, tflite_model_data):

    """

    Compare model sizes between TensorFlow and TensorFlow Lite.

   

    Args:

        tf_model_path (str): Path to TensorFlow model

        tflite_model_data (bytes): TensorFlow Lite model data

    """

    # TODO: Calculate file sizes and compression ratio
    model_size = os.stat(tf_model_path).st_size
    tflite_model_size = len(tflite_model_data)
    compression_ratio = tflite_model_size/model_size
    # TODO: Print comparison results
    print(f"original model size = {model_size} bytes")
    print(f"tflite_model size = {tflite_model_size} bytes")
    print(f"compression ratio = {compression_ratio:.2f}")

    return

 

def test_tflite_accuracy(tflite_model_data, x_test, y_test):

    """

    Test TensorFlow Lite model accuracy.

   

    Args:

        tflite_model_data (bytes): TensorFlow Lite model

        x_test, y_test: Test data

       

    Returns:

        float: Test accuracy

    """

    # TODO: Create TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_data)
    interpreter.allocate_tensors()

    # TODO: Run inference on test data
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"input_type:{input_details['dtype']}, Quant:{input_details['quantization']}")
    print(f"output_type:{output_details['dtype']}, Quant:{output_details['quantization']}")
    correct_predictions = 0
    
    # Check if the model is quantized
    is_quantized = input_details['dtype'] == np.uint8
    
    if is_quantized:
        input_scale, input_zero_point = input_details["quantization"]
        
    for i in range(x_test.shape[0]):
        input_data = x_test[i:i+1]
        if is_quantized:
            input_data = input_data / input_scale + input_zero_point
            input_data = input_data.astype(np.uint8)
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
        predicted_class = tf.argmax(output_data, axis=1).numpy()[0]
        if predicted_class == y_test[i]:
            correct_predictions += 1

    # TODO: Calculate and return accuracy
    accuracy = correct_predictions / x_test.shape[0]
    return accuracy

 

if __name__ == "__main__":

    # Convert model without quantization

    tflite_model = convert_to_tflite('mnist_cnn_model.keras', quantize=False)

   

    # Convert model with quantization

    tflite_quantized_model = convert_to_tflite('mnist_cnn_model.keras', quantize=True)

   

    # Save TensorFlow Lite models

    with open('mnist_model.tflite', 'wb') as f:

        f.write(tflite_model)

   

    with open('mnist_model_quantized.tflite', 'wb') as f:

        f.write(tflite_quantized_model)

   

    # Analyze model sizes

    analyze_model_size('mnist_cnn_model.keras', tflite_model)

    analyze_model_size('mnist_cnn_model.keras', tflite_quantized_model)

   

    # Test accuracy of converted models

    # (You'll need to load test data again)

    x_train, y_train, x_test, y_test = load_and_preprocess_data()

   

    tflite_accuracy = test_tflite_accuracy(tflite_model, x_test, y_test)

    tflite_quantized_accuracy = test_tflite_accuracy(tflite_quantized_model, x_test, y_test)

   

    print(f"TensorFlow Lite accuracy: {tflite_accuracy:.4f}")

    print(f"TensorFlow Lite quantized accuracy: {tflite_quantized_accuracy:.4f}")