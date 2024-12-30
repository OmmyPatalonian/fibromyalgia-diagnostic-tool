import tensorflow as tf

print("Starting TensorFlow test script...")

try:
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"An error occurred: {e}")