# MicroCrystalNet-Advanced-Microcrystal-Classification-Using-Normalized-Sparse-Reduction-Blocks

# README.md

## Project Overview

This repository contains code and models for a convolutional neural network (CNN) designed for classification of microcrystals in SEM images. The repository includes various scripts for data preprocessing, model training, evaluation, and results visualization.

## Repository Structure

### Model Files
- **CNN_2023-11-19_11-47-51_run1.h5**: Trained CNN model file.
- **CNN_2023-11-19_11-47-51_run1_0.001_16_Adam.h5**: Model checkpoint file with specific training parameters.
- **CNN_2023-11-19_11-47-51_run1_summary.txt**: Summary of the training process, including metrics.
- **CNN_2023-11-19_11-47-51_run1_test_predictions.npy**: Predictions on the test data.
- **CNN_2023-11-19_11-47-51_run1_weights.h5**: Model weights file.

### Python Scripts
- **Class_Balancer.py**: Script for balancing the classes in the dataset.
- **Class_Sifter.py**: Script for sifting through the dataset and segregating classes.
- **Labeling_checker.py**: Script for checking and validating data labels.
- **Model_Tester.py**: Script for testing the trained model on test data.
- **Single_CNN_Model.py**: Script containing the CNN architecture and training routine.
- **Train_Test_Splitter.py**: Script for splitting the dataset into training and testing sets.
- **merge_results.py**: Script for merging and analyzing model results.

### Data Files
- **Test Data.7z**: Compressed test dataset.
- **Validation Data.7z**: Compressed validation dataset.

### Visualization Files
- **hybrid_training_plot_CNN_2023-11-19_11-47-51.png**: Training process visualization plot.
- **t-sne_2d_kde_{LR}_{BC}_{OPT}.png**: 2D t-SNE plot with KDE visualization.
- **t-sne_2d_scatter_{LR}_{BC}_{OPT}.png**: 2D t-SNE scatter plot.
- **tsne_embeddings_{LR}_{BC}_{OPT}.npy**: t-SNE embeddings in numpy format.

## Getting Started

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required Python packages:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### Data Preparation
1. Extract the test and validation datasets:
   ```sh
   7z x "Test Data.7z" -o"./test_data"
   7z x "Validation Data.7z" -o"./validation_data"
   ```

2. Run the data splitting script:
   ```sh
   python Train_Test_Splitter.py
   ```

#### Model Training
1. Train the CNN model using the training script:
   ```sh
   python Single_CNN_Model.py
   ```

2. Check the training summary and visualize the training process:
   - Refer to the `CNN_2023-11-19_11-47-51_run1_summary.txt` for detailed metrics.
   - View `hybrid_training_plot_CNN_2023-11-19_11-47-51.png` for the training plot.

#### Model Evaluation
1. Evaluate the trained model:
   ```sh
   python Model_Tester.py
   ```

2. Analyze the test predictions:
   - Load `CNN_2023-11-19_11-47-51_run1_test_predictions.npy` for detailed predictions.

#### Visualization
1. Explore t-SNE visualizations:
   - KDE: `t-sne_2d_kde_{LR}_{BC}_{OPT}.png`
   - Scatter: `t-sne_2d_scatter_{LR}_{BC}_{OPT}.png`
   - Embeddings: `tsne_embeddings_{LR}_{BC}_{OPT}.npy`

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact
For any questions or inquiries, please contact [your email].

---

This README provides a comprehensive overview of the repository, guiding users through installation, usage, and contribution processes.
