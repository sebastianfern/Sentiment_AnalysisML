import tensorflow as tf
import numpy as np
import os

# Function to evaluate the TFLite model
def evaluate_tflite_model(tflite_model, test_data, test_labels):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct_predictions = 0
    
    # Convert the test_data to FLOAT32 before setting tensor
    test_data = test_data.astype(np.float32)
    
    for i in range(len(test_data)):
        interpreter.set_tensor(input_details[0]['index'], [test_data[i]])
        interpreter.invoke()
        
        prediction = interpreter.get_tensor(output_details[0]['index'])
        if prediction >= 0.5 and test_labels[i] == 1:
            correct_predictions += 1
        elif prediction < 0.5 and test_labels[i] == 0:
            correct_predictions += 1
    
    return correct_predictions / len(test_labels)

def get_directory_size(directory):
    total = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

