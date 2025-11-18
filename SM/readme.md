# Example CUHK-X Image Classification Model Training Script

## Overview

This example project provides a complete script for training image classification models with example data. You can download the data from the link:     
The script supports multiple data modalities (e.g., RGB, Depth, Infrared) and offers flexible data loading, preprocessing, training, and evaluation functionalities.

## Features

- Supports various network architectures (e.g., ResNet and Vision Transformer).
- Automatic handling of class imbalance issues (optional oversampling).
- Detailed logging to monitor the training process.
- Supports cross-user and intra-split data partitioning modes.

## Installation

Ensure you have Python 3.6 or higher installed, and then install the required libraries:

```bash
pip install numpy torch torchvision tqdm scikit-learn Pillow
```

## Dataset Preparation

Make sure your dataset directory structure is as follows:

```
dataset_root/
    ├── RGB/
    │   ├── label1/
    │   │   ├── user1/
    │   │   │   ├── sequence1/
    │   │   │   │   ├── image1.jpg
    │   │   │   │   └── image2.jpg
    │   │   │   └── sequence2/
    │   │   └── user2/
    │   └── label2/
    └── ...
```

## Usage

Run the following command to start training:

```bash
python train_models_cross_multi.py \
  --dataset_root /path/to/dataset \  # Root directory of the dataset
  --data rgb \                        # Data modality: rgb, depth, ir, thermal
  --epochs 15 \                       # Number of training epochs
  --gpu 0 \                           # GPU device number to use
  --network resnet50 \                # Network architecture: resnet18, resnet34, resnet50, vit_b_16
  --weights pretrained \               # Weight initialization: pretrained or scratch
  --batch_size 64 \                   # Batch size for training
  --learning_rate 0.001 \             # Learning rate for the optimizer
  --split_mode cross \                # Data splitting mode: cross or intra
  --oversample \                       # Enable minority class oversampling
  --labels "10,30" \                  # Label frequency rank range
  --log_dir /path/to/log_dir \        # Log output directory
  --cross_user_id 5                   # Test user ID in cross mode
```

## Parameter Description

- `--dataset_root`: Path to the root directory containing image data.
- `--data`: Select the data modality (e.g., RGB, Depth, Infrared).
- `--epochs`: Number of training epochs.
- `--gpu`: GPU device number to use.
- `--network`: Choose the network architecture (e.g., ResNet or ViT).
- `--weights`: Specify weight initialization method (pretrained or scratch).
- `--batch_size`: Batch size for each training iteration.
- `--learning_rate`: Learning rate for the optimizer.
- `--split_mode`: Choose data splitting mode (cross-user or random split).
- `--oversample`: Enable oversampling for minority classes.
- `--labels`: Specify label frequency rank range, supports formats like '10,30' or '0' or 'all'.
- `--log_dir`: Directory for saving log files.
- `--cross_user_id`: Specify the user ID for testing in cross mode.

## Logging

Logs generated during training will be saved in the specified `log_dir`, facilitating analysis and debugging.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please contact [syjiang@ie.cuhk.edu.hk].
```

### Explanation of Sections

- **Overview**: Briefly describes the purpose and functionality of the project.
- **Features**: Lists the main functionalities offered by the script.
- **Installation**: Provides commands for installing the necessary libraries.
- **Dataset Preparation**: Describes the required directory structure for the dataset.
- **Usage**: Offers a command to run the script, including comments for each parameter.
- **Parameter Description**: Explains the purpose of each command-line argument.
- **Logging**: Details how logs are handled during training.
- **License**: States the licensing information for the project.
- **Contribution**: Invites contributions and explains how to contribute.
- **Contact**: Provides contact information for users to reach out with questions or feedback.

Feel free to adjust the content to better fit your project’s needs or personal preferences!
