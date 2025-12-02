# CUHK-X Skeleton-based Action Recognition Training

## Overview

This project uses the DSTformer model for skeleton-based action recognition training. It supports multiple data splitting modes, including cross-trial (random 80/20 split), cross-subject (leave-one-user-out), and resampled_cross-subject-trial.

## Requirements

Ensure Python 3.9+ and required dependencies are installed:

```bash
pip install -r requirements.txt
```

Main dependencies:
- PyTorch
- sklearn
- tqdm
- numpy
- wandb (for training logging)

## Usage

### Step 1: Navigate to the skeleton directory

```bash
cd YOUR/PATH/TO/SM/Code/skeleton
```

### Step 2: Configure the number of action classes

Modify the `action_classes` parameter in the configuration file under `configs/` directory.

Edit `configs/dstformer.yaml`:

```yaml
# Data
dataset: cuhkx
action_classes: 44  # Adjust based on your actual dataset
```

**Note**: If using a subset of data (e.g., specific users or actions), adjust the number of classes accordingly.

### Step 3: Generate data split files

Modify the data paths in `split_data.py`:

```python
# Lines 6-7, modify to your dataset paths
root_dir = '/path/to/your/SM_data/Skeleton'  # Skeleton data root directory
save_dir = '/path/to/split_data_results'      # Output directory for split files
```

Run the data splitting script:

```bash
python split_data.py
```

The script will generate the following files in the `split_data_results/` directory:

#### Cross-trial mode (Random 80/20 split)
- `cross_trial_train.txt` - Training set (80%)
- `cross_trial_test.txt` - Test set (20%)
- `cross_trial_distribution.png` - Data distribution visualization

#### Cross-subject mode (Leave-one-user-out)
- `cross_subject_train_{user_id}.txt` - Training set (other 29 users)
- `cross_subject_test_{user_id}.txt` - Test set (specified user)
- `cross_subject_distribution_{user_id}.png` - Data distribution visualization

#### Cross-subject-resample mode (Long-tail removal with selected users)
- `cross_subject_trial_train_{user_range}.txt` - Training set (resampled)
- `cross_subject_trial_test_{user_range}.txt` - Test set
- `cross_subject_trial_{user_range}_distribution.png` - Data distribution visualization

### Step 4: Train the model

Train using the generated split files:

#### Cross-trial mode training

```bash
CUDA_VISIBLE_DEVICES=4,6 python train.py \
  --train_dir cross_trial_train.txt \
  --test_dir cross_trial_test.txt \
  --config ./configs/dstformer.yaml
```

#### Cross-subject mode training (All 30 users)

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py \
  --train_dir cross_subject_trial_train_30users.txt \
  --test_dir cross_subject_trial_test_30users.txt \
  --config ./configs/dstformer.yaml
```

#### Cross-subject-resample mode training (20-user subset)

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py \
  --train_dir cross_subject_train_top20_test1.txt \
  --test_dir cross_subject_test_top20_test1.txt \
  --config ./configs/dstformer.yaml
```

#### Cross-subject mode training (Leave-one-out, e.g., user10)

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py \
  --train_dir cross_subject_train_10.txt \
  --test_dir cross_subject_test_10.txt \
  --config ./configs/dstformer.yaml
```

### Step 5: Model evaluation

Use the `--evaluate` flag to evaluate a trained model:

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --train_dir cross_subject_train_10.txt \
  --test_dir cross_subject_test_10.txt \
  --config configs/dstformer.yaml \
  --evaluate checkpoint/cuhkx_cross_subject/DSTformer_cross_subject_train_10/best_epoch.bin
```

## Training Parameters

### Command-line Arguments

- `--train_dir`: Training split filename (located in split_data_results/)
- `--test_dir`: Test split filename (located in split_data_results/)
- `--config`: Model configuration file path
- `--evaluate`: Evaluation mode, specify checkpoint path
- `--resume`: Resume training from checkpoint
- `--checkpoint`: Checkpoint save directory

### Configuration File Parameters (dstformer.yaml)

```yaml
# Training parameters
epochs: 100              # Number of training epochs
batch_size: 64          # Batch size
lr_backbone: 0.00001    # Backbone learning rate
lr_head: 0.0001         # Classification head learning rate
weight_decay: 0.01      # Weight decay
lr_decay: 0.95          # Learning rate decay

# Model parameters
backbone: DSTformer     # Backbone architecture
num_joints: 17          # Number of skeleton joints
dim_feat: 64            # Feature dimension
dim_rep: 64             # Representation dimension
depth: 1                # Transformer depth
num_heads: 2            # Number of attention heads
hidden_dim: 256         # Hidden layer dimension
dropout_ratio: 0.5      # Dropout ratio

# Data augmentation
random_move: True              # Random movement augmentation
scale_range_train: [1, 3]     # Training scale range
scale_range_test: [2, 2]      # Test scale range
```

## Dataset Structure

Skeleton data should follow this directory structure:

```
Skeleton/
├── action1/
│   ├── user1/
│   │   ├── trial1/
│   │   │   └── predictions/
│   │   │       ├── sample1.npy
│   │   │       └── sample2.npy
│   │   └── trial2/
│   └── user2/
└── action2/
```

Each `.npy` file contains skeleton joint coordinates (shape: [num_frames, num_joints, 3]).

## Output

### Training Logs

- Automatically saved to `checkpoint/` directory
- Training curves logged with wandb (project name: cuhkx_cross_trial or cuhkx_cross_subject)
- Saves `latest_epoch.bin` and `best_epoch.bin`

### Evaluation Metrics

Training and evaluation output the following metrics:

- **Accuracy@1/5**: Top-1 and Top-5 accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **F1-Score**: Macro-averaged F1 score
- **AUC-ROC**: Area under ROC curve for multi-class classification

Performance metrics are computed separately for minority classes and majority classes.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please contact [syjiang@ie.cuhk.edu.hk]
