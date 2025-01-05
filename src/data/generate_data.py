import numpy as np
import pandas as pd

def generate_realistic_data(num_samples=1000):
    """
    Generate realistic HRV, GSR, and EMG data for normal or Fibromyalgia patients.
    :param num_samples: Number of samples.
    :return: Combined dataset (HRV, GSR, EMG, Condition).
    """
    print("Generating realistic data...")

    # Generate synthetic data for "fibro" and "normal" conditions
    fibro_data = np.random.normal(loc=[55, 6, 9], scale=[5, 0.5, 1], size=(num_samples // 2, 3))
    normal_data = np.random.normal(loc=[80, 3, 4], scale=[10, 1, 2], size=(num_samples // 2, 3))

    # Create DataFrame and assign labels
    fibro_df = pd.DataFrame(fibro_data, columns=['HRV', 'GSR', 'EMG'])
    fibro_df['Condition'] = 1  # 1 for Fibro

    normal_df = pd.DataFrame(normal_data, columns=['HRV', 'GSR', 'EMG'])
    normal_df['Condition'] = 0  # 0 for Normal

    # Combine the data
    data = pd.concat([fibro_df, normal_df], ignore_index=True)
    print("Data generation completed.")
    return data