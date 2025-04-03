# E-NetSRM: Exploring Preprocessing Strategies for Steganalysis

![MSVA Pipeline](https://github.com/user-attachments/assets/28b2a361-0246-426c-aa0a-bebc1afd0827)

## Setup

Download the [ALASKA2 dataset from Kaggle](https://www.kaggle.com/competitions/alaska2-image-steganalysis)

---

## Arguments

The script accepts several command-line arguments to configure training, evaluation, and model variations:

- `--mode` (int, default: `201`): Loads a specific model architecture variation.  
- `--load` (flag): Load a pre-trained model.  
- `--save` (flag): Save the trained model after execution.  
- `--train` (flag): Enable model training.  
- `--train_n` (int, default: `60000`): Number of training samples.  
- `--epochs` (int, default: `20`): Number of training epochs.  
- `--learning_rate` (float, default: `0.0001`): Learning rate for training.  
- `--evaluate` (flag): Enable model evaluation.  
- `--evaluate_n` (int, default: `1000`): Number of evaluation samples.  
- `--batch_size` (int, default: `32`): Batch size for both training and evaluation.  
- `--cover_dist_percentage` (float, default: `0.5`): Percentage of cover images in the dataset.  
- `--target_alg` (str, default: `None`): Target steganography algorithm for analysis.  
- `--group_by_base_image` (flag): Group images by base image during processing.  
