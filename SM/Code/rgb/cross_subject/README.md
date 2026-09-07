# CUHK-X RGB Cross-Subject Action Recognition Training

## Overview

This directory contains training scripts for RGB-based cross-subject action recognition, supporting multiple training strategies:
- **Supervised Learning**: Pure supervised classification training
- **Contrastive Learning**: SimCLR-based self-supervised + supervised hybrid training
- **Environment-Aware Training**: Data partitioning strategy considering environmental differences (As the data statistics for 30 users were completed in two different environments, users 1-15 belong to the first environment, while users 16-30 belong to the second environment)

## Directory Structure

```
cross_subject/
├── Training Scripts (Python)
│   ├── train_supervised_44.py          # 44-class supervised learning
│   ├── train_supervised_lt.py          # Subset action supervised learning
│   ├── simclr_44.py                    # 44-class SimCLR contrastive learning
│   ├── simclr_10.py                    # 10-class SimCLR (cross-environment)
│   ├── simclr_10_remove_env.py         # 10-class SimCLR (environment-aware)
│   └── train_models_cross_multi.py     # General multi-modal training framework
│
├── Batch Scripts (Bash)
│   ├── train_supervised_44.sh          # Batch run 44-class supervised training
│   ├── train_supervised_a.sh           # Batch run subset action supervised training
│   ├── train_contra_all_users_44.sh    # Batch run 44-class contrastive learning
│   ├── train_10_users_contra.sh        # Batch run 10-class contrastive (cross-env)
│   ├── train_10_users_contra_remove_env.sh  # Batch run 10-class contrastive (env-aware)
│   └── train_models_multi_cross.sh     # General multi-modal training launcher
│
└── Log Directories (auto-generated)
    ├── logs_supervised_all_subject/         # 44-class supervised training logs
    ├── supervised_logs_lt/                  # Subset action supervised training logs
    ├── contra_allusers_logs/                # 44-class contrastive learning logs
    ├── training_logs_contra_lt/             # 10-class contrastive logs (cross-env)
    └── training_logs_contra_lt_remove_env/  # 10-class contrastive logs (env-aware)
```

## Training Methods

### 1. Supervised Learning

Pure supervised classification training using ResNet18 as the backbone network.

#### 1.1 Full 44-Class Supervised Learning

**Script**: `train_supervised_44.py`  
**Batch Script**: `train_supervised_44.sh`

**Training Strategy**:
- **Training Set**: All 44-class action data from other 29 users
- **Test Set**: All 44-class action data from the specified test user
- **Network**: ResNet18 (ImageNet pretrained)
- **Optimizer**: Adam (lr=1e-4)
- **Data Augmentation**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter

**Usage**:

```bash
# Single user training
python train_supervised_44.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --image_size 224

# Batch training for multiple users
bash train_supervised_44.sh
```

**Configuration**:
- Edit `USERS` variable in `train_supervised_44.sh` to select test users
- Default test users: `user1, user7, user13, user20, user26`
- GPU settings: Modify `export CUDA_VISIBLE_DEVICES` in the script

**Output**:
- Log directory: `logs_supervised_all_subject/`
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score

---

#### 1.2 Subset Action Supervised Learning

**Script**: `train_supervised_lt.py`  
**Batch Script**: `train_supervised_lt.sh`

**Training Strategy**:
- **Training Set**: Specified K-class action data from other 29 users
- **Test Set**: Same K-class action data from the specified test user
- **Label Mapping**: Uses local labels (0 to K-1)
- **Default Actions**: 6, 7, 9, 11, 12, 20, 21, 32, 36, 37 (10 common actions)

**Usage**:

```bash
# Single user training
python train_supervised_lt.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --image_size 224 \
    --supervised_action_ids 6 7 9 11 12 20 21 32 36 37

# Batch training
bash train_supervised_lt.sh
```

**Key Parameters**:
- `--supervised_action_ids`: List of action IDs (space-separated)
- `--exact_match`: Use exact matching mode (recommended)

**Output**:
- Log directory: `supervised_logs_lt/`

---

### 2. Contrastive Learning (SimCLR)

Hybrid training strategy combining self-supervised contrastive learning with supervised classification.

#### 2.1 Full 44-Class Contrastive Learning

**Script**: `simclr_44.py`  
**Batch Script**: `train_contra_all_users_44.sh`

**Training Strategy**:
- **Supervised Branch**: Other 29 users × 44 actions (labeled)
- **Contrastive Branch**: All data from test user (unlabeled, SimCLR contrastive learning)
- **Loss Function**: `Loss = Loss_supervised + Loss_contrastive`
- **Network Structure**: ResNet18 + Projection Head (512→128)

**Usage**:

```bash
# Single user training
python simclr_44.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --temperature 0.5 \
    --image_size 224

# Batch training
bash train_contra_all_users_44.sh
```

**Key Parameters**:
- `--temperature`: NT-Xent loss temperature parameter (default 0.5)
- `--num_users`: Total number of users (default 30)

**Contrastive Learning Data Augmentation**:
```python
RandomResizedCrop(scale=(0.5, 1.0))
RandomHorizontalFlip()
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)
RandomGrayscale(p=0.2)
GaussianBlur(kernel_size, sigma=(0.1, 2.0))
```

**Output**:
- Log directory: `contra_allusers_logs/`

---

#### 2.2 Subset Action Contrastive Learning (Cross-Environment Version)

**Script**: `simclr_10.py`  
**Batch Script**: `train_10_users_contra.sh`

**Training Strategy**:
- **Supervised Branch**: Other 29 users (cross-environment) × specified K actions
- **Contrastive Branch**: All data from test user
- **Test Set**: Specified K-class action data from test user
- **Environment**: Does not distinguish between Environment A (user1-15) and Environment B (user16-30)

**Usage**:

```bash
# Single user training
python simclr_10.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --temperature 0.5 \
    --image_size 112 \
    --supervised_action_ids 6 7 9 11 12 20 21 32 36 37

# Batch training
bash train_10_users_contra.sh
```

**Data Scale**:
- Training data: ~116,000 samples (29 users × 10 actions × ~400 samples/action)
- Contrastive data: ~4,400 samples (1 user × 10 actions × ~400 samples/action)

**Output**:
- Log directory: `training_logs_contra_lt/`

---

#### 2.3 Subset Action Contrastive Learning (Environment-Aware Version)

**Script**: `simclr_10_remove_env.py`  
**Batch Script**: `train_10_users_contra_remove_env.sh`

**Training Strategy** (Key Improvement):
- **Environment-Aware Partitioning**: 
  - Environment A: user1-15
  - Environment B: user16-30
- **Training Data**: Only uses data from 14 users in the same environment
- **Action Intersection Calculation**: Training actions = Specified actions ∩ Actions actually performed by test user
- **Loss Weights**: `Loss = 1.0 × Loss_supervised + 0.5 × Loss_contrastive`

**Usage**:

```bash
# Single user training
python simclr_10_remove_env.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --temperature 0.5 \
    --image_size 224 \
    --supervised_action_ids 6 7 9 11 12 20 21 32 36 37 \
    --exact_match

# Batch training
bash train_10_users_contra_remove_env.sh
```

**Key Features**:
- `get_environment_users()`: Automatically identifies user's environment
- `get_user_executed_actions()`: Detects actions actually performed by user
- Action intersection validation: Ensures test set doesn't contain empty categories

**Data Scale Comparison**:
| Version | Training Users | Samples | Environment Isolation |
|---------|---------------|---------|----------------------|
| simclr_10.py | 29 (cross-env) | ~116K | ❌ |
| simclr_10_remove_env.py | 14 (same-env) | ~56K | ✅ |

**Output**:
- Log directory: `training_logs_contra_lt_remove_env/`

---

### 3. General Multi-Modal Training Framework

**Script**: `train_models_cross_multi.py`  
**Batch Script**: `train_models_multi_cross.sh`

**Training Strategy**:
- Supports multiple data modalities: RGB, Depth, IR, Thermal
- Supports multiple network architectures: ResNet18/34/50, ViT-B/16
- Supports both cross-subject and intra-split partitioning modes
- Automatic class imbalance handling (optional oversampling)

**Usage**:

```bash
# Edit configuration
vim train_models_multi_cross.sh
# Modify the following parameters:
DATASET_ROOT="/path/to/data"
DATA="rgb"              # rgb, depth, ir, thermal
NETWORK="resnet50"      # resnet18, resnet34, resnet50, vit_b_16
SPLIT_MODE="cross"      # cross, intra
CROSS_USER_ID=5         # Test user ID (cross mode only)
LABELS="all"            # "all" or "10,30" (frequency ranking range)
OVERSAMPLE_FLAG=""      # Set to "--oversample" to enable oversampling

# Run training
bash train_models_multi_cross.sh
```

**Key Parameters**:
- `--split_mode cross`: Leave-one-subject-out (LOSO)
- `--split_mode intra`: 80/20 random split
- `--labels "10,30"`: Use actions ranked 10-30 by frequency
- `--oversample`: Enable minority class oversampling
- `--weights pretrained`: Use ImageNet pretrained weights

**Output**:
- Log directory: `logs_baseline/{data}_{split_mode}_user{id}_{timestamp}/`
- Auto-generated training summary: `experiment_summary.txt`

---

## Quick Start Guide

### Step 1: Environment Setup

```bash
# Enter working directory
cd ./rgb/cross_subject

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Preparation

Ensure the dataset structure is as follows:

```
RGB/
├── action1/
│   ├── user1/
│   │   ├── trial1/
│   │   │   ├── image001.jpg
│   │   │   └── image002.jpg
│   │   └── trial2/
│   └── user2/
└── action2/
```

### Step 3: Select Training Strategy

Choose the appropriate script based on your experimental needs:

```bash
# 1. Quick baseline test (44-class supervised learning)
bash train_supervised_44.sh

# 2. Few-shot learning scenario (10-class supervised learning)
bash train_supervised_a.sh

# 3. Semi-supervised learning scenario (44-class contrastive learning)
bash train_contra_all_users_44.sh

# 4. Cross-environment generalization (10-class contrastive - cross-env)
bash train_10_users_contra.sh

# 5. Same-environment optimization (10-class contrastive - env-aware)
bash train_10_users_contra_remove_env.sh

# 6. Custom experiments (general framework)
bash train_models_multi_cross.sh
```

### Step 4: Modify Test User List

Modify the `USERS` array in bash scripts:

```bash
# Default configuration (covers both environments)
USERS=("user1" "user7" "user13" "user20" "user26")

# Test all users in Environment A (user1-15)
USERS=($(seq -f "user%g" 1 15))

# Test all users in Environment B (user16-30)
USERS=($(seq -f "user%g" 16 30))

# Test all 30 users
USERS=($(seq -f "user%g" 1 30))
```

### Step 5: GPU Configuration

Modify GPU device:

```bash
# In bash script
export CUDA_VISIBLE_DEVICES=2

# Or set temporarily in command line
CUDA_VISIBLE_DEVICES=0,1 bash train_supervised_44.sh
```

---

## Detailed Training Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--test_user` | str | required | Test user ID (e.g., user1) |
| `--dataset_path` | str | required | RGB dataset root directory |
| `--batch_size` | int | 64 | Batch size |
| `--epochs` | int | 20 | Number of training epochs |
| `--lr` | float | 1e-4 | Learning rate |
| `--image_size` | int | 224 | Image size |

### SimCLR-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--temperature` | float | 0.5 | NT-Xent loss temperature parameter |
| `--supervised_action_ids` | int+ | required | List of supervised action IDs |
| `--exact_match` | flag | False | Use exact matching mode |
| `--num_users` | int | 30 | Total number of users |

### Multi-Modal Framework Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data` | str | rgb | Data modality (rgb/depth/ir/thermal) |
| `--network` | str | resnet50 | Network architecture |
| `--weights` | str | pretrained | Weight initialization (pretrained/scratch) |
| `--split_mode` | str | cross | Data split mode (cross/intra) |
| `--cross_user_id` | int | required | Test user ID (cross mode only) |
| `--labels` | str | all | Action frequency range (e.g., "10,30") |
| `--oversample` | flag | False | Enable minority class oversampling |

---

## Experimental Results Analysis

### Log File Format

Each training script generates timestamped logs:

```
{log_dir}/{test_user}_{timestamp}.log
```

Log contents include:
```
Epoch [1/20] Train Loss: 2.3456 Train Acc: 45.67%
Epoch [1/20] Test Loss: 2.1234 Test Acc: 52.34%
Precision: 0.5123, Recall: 0.5234, F1: 0.5178
```

### Model Save Locations

```
models/
├── user1_supervised_44.pth          # 44-class supervised model
├── user1_supervised_10.pth          # 10-class supervised model
├── user1_simclr_44.pth              # 44-class contrastive learning model
├── user1_simclr_10.pth              # 10-class contrastive model (cross-env)
└── user1_simclr_env_aware.pth       # 10-class contrastive model (env-aware)
```

### Batch Results Statistics

Use the following commands to extract final accuracy for all users:

```bash
# Extract supervised learning results
grep "Test Acc:" logs_supervised_all_subject/*.log | tail -n 5

# Extract contrastive learning results
grep "Test Acc:" training_logs_contra_lt/*.log | tail -n 5

# Generate CSV statistics report
for log in logs_supervised_all_subject/*.log; do
    user=$(basename $log | cut -d'_' -f1)
    acc=$(grep "Test Acc:" $log | tail -1 | grep -oP '\d+\.\d+%')
    echo "$user,$acc"
done > supervised_results.csv
```

---

## Training Method Comparison

| Method | Training Data | Test Data | Advantages | Use Cases |
|--------|--------------|-----------|------------|-----------|
| **train_supervised_44** | 29 users × 44 classes | 1 user × 44 classes | Simple, strong baseline | Full action recognition |
| **train_supervised_a** | 29 users × K classes | 1 user × K classes | Focus on specific classes | Specific action recognition |
| **simclr_44** | 29 users × 44 classes (labeled)<br>+ 1 user × 44 classes (unlabeled) | 1 user × 44 classes | Utilizes target domain data | Semi-supervised learning |
| **simclr_10** | 29 users × K classes (labeled)<br>+ 1 user × all (unlabeled) | 1 user × K classes | Contrastive learning enhancement | Cross-env generalization |
| **simclr_10_remove_env** | 14 users × K classes (labeled)<br>+ 1 user × all (unlabeled) | 1 user × K classes | Environment isolation, reduces distribution shift | Same-env optimization |

---

## FAQ

### Q1: How to choose the appropriate training method?

- **Full action recognition**: Use `train_supervised_44.sh` or `simclr_44.sh`
- **Few-shot / specific actions**: Use `train_supervised_a.sh`
- **Target domain with unlabeled data**: Use `simclr_10.sh` or `simclr_10_remove_env.sh`
- **Cross-environment generalization**: Use `simclr_10.sh` (29 users)
- **Same-environment optimization**: Use `simclr_10_remove_env.sh` (14 users)

### Q2: Why does the environment-aware version use less training data?

Environment A (user1-15) and Environment B (user16-30) have differences in lighting, background, etc. Cross-environment training has more data but introduces distribution shift. The environment-aware version isolates environments:
- Reduces training samples by 52% (116K → 56K)
- Eliminates environment-induced distribution differences
- Improves same-environment generalization

### Q3: Why is the ColorJitter hue parameter set to 0?

Older versions of torchvision have a bug where `hue=0.1` causes `Python integer -XX out of bounds for uint8` error. All scripts have been fixed with `hue=0`.

### Q4: How to modify contrastive learning loss weights?

Edit the loss calculation in the Python script:

```python
# simclr_10.py (original version)
loss = supervised_loss + contrastive_loss  # weight 1:1

# simclr_10_remove_env.py (environment-aware version)
loss = supervised_loss + 0.5 * contrastive_loss  # weight 1:0.5
```

### Q5: How to add new data augmentation?

Edit the `get_train_transform()` function:

```python
def get_train_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        
        # Add new augmentation
        transforms.RandomApply([
            transforms.RandomRotation(degrees=15)
        ], p=0.3),
        
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0)
        ], p=0.8),
        # ... other augmentations
    ])
```

### Q6: How to monitor training progress?

```bash
# View logs in real-time
tail -f training_logs_contra_lt/user1_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Extract accuracy curve
grep "Test Acc:" training_logs_contra_lt/user1_*.log
```

### Q7: What to do when running out of GPU memory?

```bash
# Method 1: Reduce batch_size
--batch_size 32  # default 64

# Method 2: Reduce image size
--image_size 112  # default 224

# Method 3: Use a smaller network
# Edit Python script to use lighter networks instead of ResNet18

# Method 4: Use gradient accumulation (requires code modification)
```

### Q8: How to export experimental results table?

```bash
# Create result extraction script
cat > extract_results.sh << 'EOF'
#!/bin/bash
echo "User,Method,Accuracy,Precision,Recall,F1" > results.csv
for log in logs_supervised_all_subject/*.log; do
    user=$(basename $log | cut -d'_' -f1)
    acc=$(grep "Test Acc:" $log | tail -1 | grep -oP '\d+\.\d+')
    prec=$(grep "Precision:" $log | tail -1 | grep -oP 'Precision: \K\d+\.\d+')
    rec=$(grep "Recall:" $log | tail -1 | grep -oP 'Recall: \K\d+\.\d+')
    f1=$(grep "F1:" $log | tail -1 | grep -oP 'F1: \K\d+\.\d+')
    echo "$user,Supervised_44,$acc,$prec,$rec,$f1" >> results.csv
done
EOF

chmod +x extract_results.sh
./extract_results.sh
```

---

## Advanced Usage

### Custom Action Subsets

```bash
# RGB common 10 actions (default)
SUP_ACTIONS="6 7 9 11 12 20 21 32 36 37"

# Depth common 10 actions
SUP_ACTIONS="2 6 7 11 20 21 32 34 36 37"

# Custom 5 core actions
SUP_ACTIONS="6 7 9 11 12"
```

### Parallel Training for Multiple Users

```bash
# Using GNU parallel (requires installation)
parallel -j 4 python train_supervised_44.py --test_user {} ::: user{1..30}

# Or using background tasks
for user in user{1..5}; do
    CUDA_VISIBLE_DEVICES=$((i%4)) python train_supervised_44.py \
        --test_user $user &
    ((i++))
done
wait
```

### Automatic Hyperparameter Search

```bash
# Grid search for learning rate and batch_size
for lr in 1e-3 1e-4 1e-5; do
    for bs in 32 64 128; do
        python simclr_10.py \
            --test_user user1 \
            --lr $lr \
            --batch_size $bs \
            --epochs 20 \
            > logs/user1_lr${lr}_bs${bs}.log 2>&1
    done
done
```

---

## Code Architecture

### Model Structure

All scripts use a unified model architecture:

```python
class UnifiedModel(nn.Module):
    def __init__(self, num_classes, projection_dim=128):
        super().__init__()
        # Backbone network
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the last FC layer
        
        # Classification head (supervised learning)
        self.classifier = nn.Linear(512, num_classes)
        
        # Projection head (contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
```

### Loss Functions

```python
# Supervised loss
supervised_loss = nn.CrossEntropyLoss()(logits, labels)

# Contrastive loss (NT-Xent)
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.T) / temperature
    # ... see code implementation for details
    return loss
```

### Data Loading Pipeline

```
1. get_dataloaders()
   ├─> Scan dataset directory
   ├─> Build supervised data list (other users)
   ├─> Build contrastive data list (test user)
   └─> Build test data list (test user)

2. ActionDataset
   ├─> Supervised mode: returns (image, label)
   └─> Contrastive mode: returns (view1, view2)

3. DataLoader
   ├─> Batch sampling
   ├─> Data augmentation (on-the-fly)
   └─> Return batch
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{cuhkx2025,
  title={CUHK-X: Multi-modal Cross-subject Action Recognition Dataset},
  author={Your Name},
  journal={Conference/Journal},
  year={2025}
}
```

**Related Methods**:
- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## Contact

For questions or suggestions, please contact:
- Email: syjiang@ie.cuhk.edu.hk
- GitHub Issues: [Project Repository Link]

---

## Changelog

- **2025-12-02**: Created complete README documentation (English version)
- **2025-11-XX**: Fixed ColorJitter hue parameter overflow bug
- **2025-11-XX**: Added environment-aware training version (simclr_10_remove_env.py)
- **2025-10-XX**: Initial version release
