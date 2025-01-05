import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

def preprocess_data(data):
    """
    Preprocess the data by scaling features and handling imbalanced data.
    """
    # Separate fibro and normal data
    fibro_data = data[data['Condition'] == 1]
    normal_data = data[data['Condition'] == 0]
    
    # Combine data for scaling
    combined_data = pd.concat([fibro_data, normal_data], axis=0)
    
    # Use a single scaler for the entire dataset
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data[['HRV', 'GSR', 'EMG']])
    
    # Save the scaler parameters for consistency
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    
    # Split the scaled data back into fibro and normal
    fibro_data.loc[:, ['HRV', 'GSR', 'EMG']] = combined_data_scaled[:len(fibro_data)]
    normal_data.loc[:, ['HRV', 'GSR', 'EMG']] = combined_data_scaled[len(fibro_data):]
    
    # Resample to balance the dataset
    min_samples = min(len(fibro_data), len(normal_data))
    fibro_data = resample(fibro_data, replace=False, n_samples=min_samples, random_state=42)
    normal_data = resample(normal_data, replace=False, n_samples=min_samples, random_state=42)
    
    # Combine the balanced data
    data = pd.concat([fibro_data, normal_data], ignore_index=True)
    
    return data

def load_scaler():
    """
    Load the saved scaler parameters.
    """
    mean = np.load('scaler_mean.npy')
    scale = np.load('scaler_scale.npy')
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    return scaler