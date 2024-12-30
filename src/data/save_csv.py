import os
import numpy as np
import pandas as pd

def generate_and_save_csv(generator, scaler, latent_dim, num_samples, num_files):
    print("Generating and saving CSV files...")
    os.makedirs("generated_csvs", exist_ok=True)

    for i in range(num_files):
        is_fibro = (i % 4 == 0)  # Fibro in every 4th file
        condition = "fibro" if is_fibro else "normal"

        # Generate synthetic data
        noise = np.random.normal(0, 1, (num_samples, latent_dim))
        synthetic_data = generator.predict(noise)
        synthetic_data = scaler.inverse_transform(synthetic_data)

        # Save to CSV
        filename = f"generated_csvs/{condition}_dataset_{i + 1}.csv"
        df = pd.DataFrame(synthetic_data, columns=['HRV', 'GSR', 'EMG'])
        df['Condition'] = condition  # Add condition column
        df.to_csv(filename, index=False)

        print(f"Saved {condition} dataset: {filename}")
    print("CSV generation and saving completed.")