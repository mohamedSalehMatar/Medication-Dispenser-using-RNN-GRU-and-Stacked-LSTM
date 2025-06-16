# convert_keras_to_h5.py

import tensorflow as tf
import argparse
import os

def convert_keras_to_h5(input_path, output_path=None):
    # Load the .keras model
    model = tf.keras.models.load_model(input_path)

    # Determine output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + ".h5"

    # Save the model in HDF5 format
    model.save(output_path, save_format='h5')
    print(f"Model converted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .keras model file to .h5 format")
    parser.add_argument("input_path", help="Path to the .keras model file")
    parser.add_argument("--output_path", help="Optional path for the output .h5 file")
    args = parser.parse_args()

    convert_keras_to_h5(args.input_path, args.output_path)
