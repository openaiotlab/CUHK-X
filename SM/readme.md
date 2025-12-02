# Example CUHK-X Image Classification Model Training Script

## Overview

This example project provides a complete script for training image classification models with example data. You can download the data from the link:     
The script supports multiple data modalities (e.g., RGB, Depth, Infrared, and Thermal) and offers flexible data loading, preprocessing, training, and evaluation functionalities.

## Features

- Supports various network architectures (e.g., ResNet and Vision Transformer).
- Automatic handling of class imbalance issues (optional oversampling).
- Detailed logging to monitor the training process.
- Supports cross-user and intra-split data partitioning modes.

## Installation

Ensure you have Python 3.8 or higher installed, and then install the required libraries:



```bash

conda create -n cuhkx python=3.9

conda activate cuhkx

pip -r requirements.txt
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
Where the labels always mapping action name 

## Usage

Run the following command to start training:

```bash
unified training for "rgb/depth/ir/thermal"

cd YOUR/PATH/rgb
# where four modes of data training contained in this folder

python train_models_cross_multi.py \
  --dataset_root /path/to/dataset \  # Root directory of the dataset
  --data rgb \                        # Data modality: rgb, depth, ir, thermal
  --epochs 15 \                       # Number of training epochs
  --gpu 0 \                           # GPU device number to use
  --network resnet50 \                # Network architecture: resnet18, resnet34, resnet50, vit_b_16
  --weights pretrained \               # Weight initialization: pretrained or scratch
  --batch_size 64 \                   # Batch size for training
  --learning_rate 0.001 \             # Learning rate for the optimizer
  --split_mode intra \                # Data splitting mode: cross_subject or intra(80%/20%)
  --oversample \                       # Enable minority class oversampling
  --labels "10,30" \                  # Label frequency rank range or you can choose all labels
  --log_dir /path/to/log_dir \        # Log output directory
  --cross_user_id                    # Test user ID in cross_user mode

or you can adjust the train_mudels_multi_intra.sh parameter to train
bash train_models_multi_intra.sh
```

```bash
or you can directly run the bash for the rgb modes:

cd cross_subject
# 1. fast baseline cross_subject training
bash train_supervised_44.sh

# 2. resampled cross_subject training 
bash train_supervised_lt.sh

# 3. cross_subject for all actions contrastive learning 
bash train_contra_all_users_44.sh

# 4. cross_subject for resampled actions contrastive learning 
bash train_10_users_contra.sh

# 5. cross_subject for resampled actions without different env contrastive learning 
bash train_10_users_contra_remove_env.sh

 
```

```bash
# for skeleton

cd skeleton

#cross_trial/intra dataset

CUDA_VISIBLE_DEVICES=4,6 python train.py --train_dir cross_trial_train.txt --test_dir cross_trial_test.txt --config ./configs/dstformer.yaml

if you want to adjust the parameters of imu scripts, see the readme.md under folder of imu

```

```bash
# for radar

cd radar

bash ./train_radar_mix.sh

if you want to adjust the parameters of imu scripts, see the readme.md under folder of radar
```

```bash
# for imu

cd imu

# run cross trail

bash ./command_accgyrmag_transformer_crosstrail.sh

if you want to adjust the parameters of imu scripts, see the readme.md under folder of imu
```


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