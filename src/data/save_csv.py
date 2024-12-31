import os
import numpy as np
import pandas as pd

def generate_and_save_csv(generator, scaler, latent_dim, num_samples, num_files):
    print("Generating and saving CSV files...")
    os.makedirs("generated_csvs", exist_ok=True)

    for i in range(num_files):
        noise = np.random.normal(0, 1, (num_samples, latent_dim))
        labels = np.random.randint(0, 2, (num_samples, 1))
        synthetic_data = generator.predict([noise, labels])
        synthetic_data = scaler.inverse_transform(synthetic_data)

        # Save to CSV
        filename = f"generated_csvs/synthetic_data_{i}.csv"
        df = pd.DataFrame(synthetic_data, columns=['HRV', 'GSR', 'EMG'])
        df['Condition'] = labels  # Add condition column
        df.to_csv(filename, index=False)

        print(f"Saved synthetic dataset: {filename}")
    print("CSV generation and saving completed.")