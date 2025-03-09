import h5py
import numpy as np
import argparse
import sys

def print_metadata(h5file):
    """Print metadata information about the HDF5 file and its datasets."""
    print("=== HDF5 File Metadata ===")
    print(f"File: {h5file.filename}")
    print("\nFile Attributes:")
    for key, value in h5file.attrs.items():
        print(f"  {key}: {value}")

    print("\nDatasets:")
    for name, dataset in h5file.items():
        print(f"\n--- Dataset: {name} ---")
        print(f"  Shape: {dataset.shape}")
        print(f"  Data Type: {dataset.dtype}")
        print(f"  Size: {dataset.size} elements")
        print(f"  Memory Size: {dataset.nbytes / (1024*1024):.2f} MB")
        
        print("\n  Dataset Attributes:")
        for key, value in dataset.attrs.items():
            print(f"    {key}: {value}")

        # Print some basic statistics about the data
        data = dataset[:]
        print("\n  Data Statistics:")
        print(f"    Min Value: {np.min(data):.6f}")
        print(f"    Max Value: {np.max(data):.6f}")
        print(f"    Mean Value: {np.mean(data):.6f}")
        print(f"    Standard Deviation: {np.std(data):.6f}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Read and display HDF5 volume data metadata.')
    parser.add_argument('--h5_file', help='Path to the HDF5 file')
    
    # Parse arguments
    args = parser.parse_args()

    try:
        # Open the HDF5 file
        with h5py.File(args.h5_file, 'r') as f:
            print_metadata(f)
            
    except OSError as e:
        print(f"Error: Could not open file '{args.h5_file}'", file=sys.stderr)
        print(f"Details: {str(e)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
