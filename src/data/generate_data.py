import numpy as np

def generate_realistic_data(num_samples, is_fibro=False):
    """
    Generate realistic HRV, GSR, and EMG data for normal or Fibromyalgia patients.
    :param num_samples: Number of samples.
    :param is_fibro: Whether the data represents Fibromyalgia symptoms.
    :return: Combined dataset (HRV, GSR, EMG).
    """
    print(f"Generating realistic data for {'Fibromyalgia' if is_fibro else 'Normal'} patients...")
    if is_fibro:
        # Fibromyalgia ranges
        hrv_data = np.random.uniform(50, 60, size=num_samples)
        gsr_data = np.random.uniform(5.3, 7, size=num_samples)
        emg_data = np.random.uniform(8, 10, size=num_samples)
    else:
        # Normal ranges
        hrv_data = np.random.uniform(60, 100, size=num_samples)
        gsr_data = np.random.uniform(1, 5, size=num_samples)
        emg_data = np.random.uniform(1, 7, size=num_samples)

    combined_data = np.stack([hrv_data, gsr_data, emg_data], axis=1)
    print("Data generation completed.")
    return combined_data