import numpy as np
import os
import configparser
from pathlib import Path


class EmbeddingQuantizer:
    def __init__(self, config_path='config.properties'):
        self.config = self._load_config(config_path)
        self.input_file = self.config.get('Input/Output paths', 'input_file')
        self.output_dir = self.config.get('Input/Output paths', 'output_dir')
        self.use_4bit = self.config.getboolean('Quantization settings', 'use_4bit')

    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def _generate_output_filename(self):
        base_name = Path(self.input_file).stem
        bits = '4bit' if self.use_4bit else '8bit'
        return f"{base_name}_sq_{bits}.npy"

    def scalar_quantization(self, arr):
        """Unified quantization method for both 4-bit and 8-bit"""
        min_val = np.min(arr)
        max_val = np.max(arr)

        # Determine range based on bit depth
        max_range = 15 if self.use_4bit else 255

        # Scale to appropriate range
        scaled = (arr - min_val) * (max_range / (max_val - min_val))

        # Round to nearest integer and convert to appropriate type
        quantized = np.round(scaled).astype(np.uint8)

        # For 4-bit, pack two values into one byte
        if self.use_4bit:
            quantized = self._pack_4bit(quantized)

        return quantized, min_val, max_val

    def _pack_4bit(self, arr):
        """Pack two 4-bit values into one byte"""
        # Ensure even length
        if arr.shape[1] % 2 != 0:
            arr = np.pad(arr, ((0, 0), (0, 1)), mode='constant')

        # Reshape and pack
        reshaped = arr.reshape(-1, 2)
        packed = (reshaped[:, 0] << 4) | reshaped[:, 1]
        return packed.reshape(arr.shape[0], -1)

    def process(self):
        # Read the original numpy array
        input_path = os.path.join(self.output_dir, self.input_file)
        original_array = np.load(input_path)

        # Perform quantization
        quantized_array, min_val, max_val = self.scalar_quantization(original_array)

        # Generate output filename
        output_file = self._generate_output_filename()
        output_path = os.path.join(self.output_dir, output_file)

        # Save the quantized array
        np.save(output_path, quantized_array)

        # Save metadata
        metadata = {
            'min_val': min_val,
            'max_val': max_val,
            'original_shape': original_array.shape,
            'is_4bit': self.use_4bit
        }
        metadata_path = output_path.replace('.npy', '_metadata.npy')
        np.save(metadata_path, metadata)

        return {
            'original_shape': original_array.shape,
            'quantized_shape': quantized_array.shape,
            'original_dtype': original_array.dtype,
            'quantized_dtype': quantized_array.dtype,
            'output_file': output_file,
            'metadata_file': metadata_path
        }


def main():
    quantizer = EmbeddingQuantizer()
    results = quantizer.process()

    # Print results
    print(f"Original shape: {results['original_shape']}")
    print(f"Quantized shape: {results['quantized_shape']}")
    print(f"Original dtype: {results['original_dtype']}")
    print(f"Quantized dtype: {results['quantized_dtype']}")
    print(f"Quantized array saved to: {results['output_file']}")
    print(f"Metadata saved to: {results['metadata_file']}")

    # Load and print first element
    quantized_data = np.load(os.path.join(quantizer.output_dir, results['output_file']))
    print(f"First element: {quantized_data[0]}")
    print(f"First element length: {len(quantized_data[0])}")
    print(f"Total number of elements: {len(quantized_data)}")


if __name__ == "__main__":
    main()
