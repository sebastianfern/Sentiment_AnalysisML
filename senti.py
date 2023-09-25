#removing Tensorflow warnings from console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


import numpy as np
import pandas as pd
import nltk
from evaluateModel import evaluate_tflite_model, get_directory_size
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
import shutil



# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load data from CSV
data = pd.read_csv("./data/IMDB_Dataset.csv")

# Remove stopwords from the reviews
data["review"] = data["review"].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in STOPWORDS]))

reviews = data["review"].values
sentiments = data["sentiment"].map({"positive": 1, "negative": 0}).values

# Split data: 25000 for training, 25000 for testing
train_reviews = reviews[:25000]
train_labels = sentiments[:25000]

test_reviews = reviews[25000:]
test_labels = sentiments[25000:]

# Pre-processing
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_reviews)

train_sequences = tokenizer.texts_to_sequences(train_reviews)
train_padded_sequences = pad_sequences(train_sequences, maxlen=100, padding="post")

test_sequences = tokenizer.texts_to_sequences(test_reviews)
test_padded_sequences = pad_sequences(test_sequences, maxlen=100, padding="post")
test_padded_sequences = test_padded_sequences.astype(np.float32)



# Model
model = Sequential([
    Embedding(10000, 64, input_length=100),
    LSTM(32, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Train the model with early stopping
model.fit(train_padded_sequences, train_labels, epochs=10, validation_data=(test_padded_sequences, test_labels), batch_size=64, callbacks=[early_stop])

# Save the model, checking if the path exists
if os.path.exists("saved_model_path"):
    shutil.rmtree("saved_model_path")
tf.saved_model.save(model, "saved_model_path")

original_size = get_directory_size("saved_model_path")

# Checking for decrease in accuracy before and after quantization
#BEFORE
loss, accuracy = model.evaluate(test_padded_sequences, test_labels)
print(f"Original Model Test Loss: {loss}")
print(f"Original Model Test Accuracy: {accuracy:.2f}%")

# Quantize the model
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_path")
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Allow the converter to use TF operations that aren't part of the default TFLite operations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

# Disable the lowering of tensor list operations
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

quantized_size = os.path.getsize("model.tflite")

# Report the size reduction
print(f"Original Model Size: {original_size / 1024 / 1024:.2f} MB")
print(f"Quantized Model Size: {quantized_size / 1024 / 1024:.2f} MB")
reduction = ((original_size - quantized_size) / original_size) * 100
print(f"Size Reduction: {reduction:.2f}%")

# Print the shape and datatype of the test data
print(test_padded_sequences.shape, test_padded_sequences.dtype)

# Evaluate the Quantized Model
quantized_accuracy = evaluate_tflite_model(tflite_model, test_padded_sequences, test_labels)
print(f"Quantized Model Accuracy: {quantized_accuracy * 100:.2f}%")

# Sample prediction
test_text = ["I didn't like the movie."]
test_seq = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_seq, maxlen=100, padding="post")

prediction = model.predict(test_padded)
print("Sentiment:", "Positive" if prediction[0] >= 0.5 else "Negative")
