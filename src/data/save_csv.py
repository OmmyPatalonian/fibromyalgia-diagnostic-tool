import os
import numpy as np
import pandas as pd

def generate_and_save_csv(generator, scaler, latent_dim, num_samples, num_files):
    print("Generating and saving CSV files...")
    os.makedirs("generated_csvs", exist_ok=True)

    for i in range(num_files):
        noise = np.random.normal(0, 1, (num_samples, latent_dim))
        labels = np.random.randint(0, 2, (num_samples, 1))
        generated_data = generator.predict([noise, labels])
        generated_data = scaler.inverse_transform(generated_data)
        condition = ['Fibro' if label == 1 else 'Normal' for label in labels.flatten()]

        # Save to CSV
        file_path = os.path.join('generated_csvs', f'synthetic_data_{i}.csv')
        df = pd.DataFrame(generated_data, columns=['HRV', 'GSR', 'EMG'])
        df['Condition'] = condition  # Add condition column
        df.to_csv(file_path, index=False)

        print(f"Saved synthetic dataset: {file_path}")
    print("CSV generation and saving completed.")

def generate_and_save_csv(synthetic_data_df, filename, latent_dim, num_samples, num_files):
    # Save the synthetic data to a CSV file
    synthetic_data_df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")