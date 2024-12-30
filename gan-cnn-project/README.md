# GAN-CNN Project

This project generates synthetic datasets using GANs, processes them with GNNs, and analyzes them with CNNs to detect Fibromyalgia Syndrome (FMS).

## Directory Structure

- `src/gan`: Contains GAN-related code.
- `src/cnn`: Contains CNN-related code.
- `src/gnn`: Contains GNN-related code.
- `src/autoencoder`: Contains Autoencoder-related code.
- `src/data`: Contains data generation and saving code.
- `src/utils`: Contains utility functions.
- `src/main.py`: Main script to run the entire pipeline.

## Setup

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

1. **Generate Data**: Use the `generate_data.py` script to create synthetic HRV, GSR, and EMG data.
2. **Train GAN**: Run the `train_gan.py` script to train the GAN model on the generated data.
3. **Analyze Data**: Use the `analyze_and_report.py` script to analyze the generated datasets with the CNN model.

## Requirements

- Python 3.x
- numpy
- pandas
- tensorflow
- scikit-learn

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.