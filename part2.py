import tensorflow as tf
from part1 import load_and_preprocess_data
import os

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

    # TODO: Run inference on test data

    # TODO: Calculate and return accuracy

    pass

 

if __name__ == "__main__":

    # Convert model without quantization

    tflite_model = convert_to_tflite('mnist_cnn_model.h5', quantize=False)

   

    # Convert model with quantization

    tflite_quantized_model = convert_to_tflite('mnist_cnn_model.h5', quantize=True)

   

    # Save TensorFlow Lite models

    with open('mnist_model.tflite', 'wb') as f:

        f.write(tflite_model)

   

    with open('mnist_model_quantized.tflite', 'wb') as f:

        f.write(tflite_quantized_model)

   

    # Analyze model sizes

    analyze_model_size('mnist_cnn_model.h5', tflite_model)

    analyze_model_size('mnist_cnn_model.h5', tflite_quantized_model)

   

    # Test accuracy of converted models

    # (You'll need to load test data again)

    # x_train, y_train, x_test, y_test = load_and_preprocess_data()

   

    # tflite_accuracy = test_tflite_accuracy(tflite_model, x_test, y_test)

    # tflite_quantized_accuracy = test_tflite_accuracy(tflite_quantized_model, x_test, y_test)

   

    # print(f"TensorFlow Lite accuracy: {tflite_accuracy:.4f}")

    # print(f"TensorFlow Lite quantized accuracy: {tflite_quantized_accuracy:.4f}")