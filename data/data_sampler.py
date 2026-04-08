"""
This file contains code for sampling the dataset. It is intended to be used for testing and debugging purposes, as well as for creating smaller subsets of the data for faster experimentation. The code in this file is not meant to be run as part of the main data loading and preprocessing pipeline, but rather as a separate utility for working with the dataset.

*NOTE: The sampling process is random, but a fixed random seed is used to ensure reproducibility. Adjust the sample size as needed for your specific use case.*

Please also run `load.py` before running this file, as it relies on the cleaned and merged dataset saved by that script.
"""
import pandas as pd

def sample_data(input_path, output_path, n_samples=1000):
    """
    Sample a subset of the data from the input path and save it to the output path.
    
    Parameters:
    - input_path: str, path to the input CSV file containing the full dataset
    - output_path: str, path to save the sampled dataset
    - n_samples: int, number of samples to include in the output dataset. Defaults to 1000.
    """
    print(f"Loading full dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Sampling {n_samples} rows from the dataset...")
    sampled_df = df.sample(n=n_samples, random_state=42)
    
    print(f"Saving sampled dataset to {output_path}...")
    sampled_df.to_csv(output_path, index=False)
    
    print("Sampling complete.")

if __name__ == "__main__":
    sample_size = 200  # Adjust this number as needed
    input_csv = "data/all_reviews_dataset.csv"
    output_csv = f"data/sample_reviews_dataset_{sample_size}.csv"
    sample_data(input_csv, output_csv, n_samples=sample_size)