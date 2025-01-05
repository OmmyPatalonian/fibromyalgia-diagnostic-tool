import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as  plt
from scipy.linalg import sqrtm
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data

# Define the ranges for fibro and normal
FIBRO_RANGES = {
    'HRV': (50, 60),
    'GSR': (5.3, 7),
    'EMG': (8, 10)
}

NORMAL_RANGES = {
    'HRV': (60, 100),
    'GSR': (1, 5),
    'EMG': (1, 7)
}

def check_range(value, range_tuple):
    return range_tuple[0] <= value <= range_tuple[1]

def predict_condition(data):
    fibro_count = 0
    normal_count = 0

    for index, row in data.iterrows():
        if all(check_range(row[col], FIBRO_RANGES[col]) for col in FIBRO_RANGES):
            fibro_count += 1
        elif all(check_range(row[col], NORMAL_RANGES[col]) for col in NORMAL_RANGES):
            normal_count += 1

    return 'fibro' if fibro_count > normal_count else 'normal'

def preprocess_data(data):
    """
    Preprocess the data by scaling features and handling imbalanced data.
    """
    # Combine data for scaling
    combined_data = pd.concat([data['fibro'], data['normal']], axis=0)
    
    # Use a single scaler for the entire dataset
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data[['HRV', 'GSR', 'EMG']])
    
    # Split the scaled data back into fibro and normal
    data['fibro'][['HRV', 'GSR', 'EMG']] = combined_data_scaled[:len(data['fibro'])]
    data['normal'][['HRV', 'GSR', 'EMG']] = combined_data_scaled[len(data['fibro']):]
    
    # Resample to balance the dataset
    min_samples = min(len(data['fibro']), len(data['normal']))
    data['fibro'] = resample(data['fibro'], replace=False, n_samples=min_samples, random_state=42)
    data['normal'] = resample(data['normal'], replace=False, n_samples=min_samples, random_state=42)
    
    return data

def calculate_fid(real_features, generated_features):
    """
    Calculate the Frechet Inception Distance (FID) between real and generated features.
    """
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def visualize_data(real_data, generated_data):
    """
    Visualize the distributions of real and generated data.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(real_data, bins=50, alpha=0.7, label='Real Data')
    plt.title('Real Data Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(generated_data, bins=50, alpha=0.7, label='Generated Data')
    plt.title('Generated Data Distribution')
    
    plt.show()

def create_knn_graph(features, k=5):
    """
    Create a K-Nearest Neighbors graph from the features.
    """
    edge_index = knn_graph(features, k=k)
    return edge_index

def analyze_and_report(cnn_model, gnn_features, ideal_values, condition_label):
    """
    Analyze the GNN features using the CNN and compare with ideal FMS profile.
    """
    print("Analyzing and reporting...")
    cnn_model.eval()
    with torch.no_grad():
        cnn_output = cnn_model(gnn_features)

    # Log intermediate values
    print(f"CNN Output Shape: {cnn_output.shape}")
    print(f"CNN Output: {cnn_output}")

    # Convert ideal_values to a tensor
    ideal_tensor = torch.tensor(ideal_values, dtype=torch.float).unsqueeze(0)

    # Ensure dimensions are compatible for cosine similarity
    if cnn_output.shape[1] != ideal_tensor.shape[1]:
        # Adjust dimensions of ideal_tensor to match cnn_output
        ideal_tensor = torch.nn.functional.pad(ideal_tensor, (0, cnn_output.shape[1] - ideal_tensor.shape[1]), "constant", 0)

    # Log intermediate values
    print(f"Ideal Tensor Shape: {ideal_tensor.shape}")
    print(f"Ideal Tensor: {ideal_tensor}")

    similarity = cosine_similarity(cnn_output.detach().numpy(), ideal_tensor.detach().numpy())[0][0] * 100
    print(f"Similarity: {similarity}%")

    # Adjust threshold dynamically based on F1 score
    y_true = [1 if condition_label == 'fibro' else 0]
    y_scores = [similarity]
    unique_labels, counts = np.unique(y_true, return_counts=True)
    print("Labels distribution:", dict(zip(unique_labels, counts)))

    if len(unique_labels) > 1:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        print("Optimal Threshold:", optimal_threshold)
    else:
        optimal_threshold = 0.5  # Default threshold

    if similarity > optimal_threshold:
        prediction = 'fibro'
    else:
        prediction = 'normal'

    print(f"Condition: {condition_label}, Prediction: {prediction}, Similarity: {similarity}%")

    # Print classification report
    print(classification_report(y_true, [1 if similarity > optimal_threshold else 0]))

def process_csvs(csv_dir, max_files=5):
    """
    Processes a limited number of CSV files in the specified directory, printing predictions,
    actual conditions, and calculating accuracy.

    Parameters:
    - csv_dir: Directory containing labeled CSV files.
    - max_files: Maximum number of CSV files to process for debugging.
    """
    correct_predictions = 0
    total_predictions = 0
    processed_files = 0

    print("Processing CSV files...\n")
    print(f"{'File':<20} {'Actual Condition':<20} {'Predicted Condition':<20} {'Correct':<10}")
    print("-" * 70)

    for file_name in os.listdir(csv_dir):
        if processed_files >= max_files:
            break

        if file_name.endswith(".csv"):
            # Extract condition from file name
            if "fibro" in file_name.lower():
                actual_condition = "fibro"
            elif "normal" in file_name.lower():
                actual_condition = "normal"
            else:
                actual_condition = "unknown"

            # Load CSV and make predictions
            file_path = os.path.join(csv_dir, file_name)
            data = pd.read_csv(file_path)
            
            # Debug: Print the first few rows of the data
            print(f"\nCSV Data Sample from {file_name}:")
            print(data.head())

            # Check for NaN values in the Condition column
            if data['Condition'].isnull().any():
                print(f"Skipping {file_name} due to NaN values in Condition column.")
                continue

            predicted_condition = predict_condition(data)
            correct = (predicted_condition == actual_condition)
            correct_predictions += correct
            total_predictions += 1
            processed_files += 1

            print(f"{file_name:<20} {actual_condition:<20} {predicted_condition:<20} {str(correct):<10}")

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}%")

# Example usage
if __name__ == "__main__":
    csv_directory = "path/to/generated_csvs"
    process_csvs(csv_directory, max_files=5)