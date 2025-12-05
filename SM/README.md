<div align="center">

# 🎯 CUHK-X Multi-Modal Action For Small Model Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUHK](https://img.shields.io/badge/CUHK-Research-purple.svg)](https://www.cuhk.edu.hk/)

**A comprehensive framework for multi-modal action recognition supporting RGB, Depth, Infrared, Thermal, Skeleton, Radar, and IMU data.**

[📖 Overview](#-overview) •
[✨ Features](#-features) •
[🚀 Quick Start](#-quick-start) •
[📁 Dataset](#-dataset-preparation) •
[💻 Usage](#-usage) •
[📧 Contact](#-contact)

</div>

---

## 📖 Overview

This project provides a complete training pipeline for **multi-modal action recognition** models. It supports various data modalities and offers flexible data loading, preprocessing, training, and evaluation functionalities.

> 📥 **Dataset Download**: [Coming Soon]

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏗️ **Multiple Architectures** | ResNet (18/34/50), Vision Transformer (ViT) |
| 📊 **Multi-Modal Support** | RGB, Depth, Infrared, Thermal, Skeleton, Radar, IMU |
| ⚖️ **Class Imbalance Handling** | Optional oversampling for minority classes |
| 📝 **Comprehensive Logging** | Detailed training process monitoring |
| 🔀 **Flexible Data Splitting** | Cross-user and intra-split partitioning modes |
| 🎯 **Contrastive Learning** | Support for self-supervised pre-training |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)

### Installation

```bash
# Create conda environment
conda create -n cuhkx python=3.9
conda activate cuhkx

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

Organize your dataset with the following structure:

```
dataset_root/
├──  RGB/
│   ├──  label1/
│   │   ├──  user1/
│   │   │   ├──  sequence1/
│   │   │   │   ├──  image1.jpg
│   │   │   │   └──  image2.jpg
│   │   │   └──  sequence2/
│   │   └──  user2/
│   └──  label2/
├──  Depth/
├──  IR/
├──  Thermal/
└── ...
```

> 💡 **Note**: Labels correspond to action names.

---

## 💻 Usage

### 🎨 RGB / Depth / IR / Thermal Training

<details>
<summary><b>🔧 Option 1: Command Line</b></summary>

```bash
cd YOUR/PATH/rgb

python train_models_cross_multi.py \
  --dataset_root /path/to/dataset \
  --data rgb \
  --epochs 15 \
  --gpu 0 \
  --network resnet50 \
  --weights pretrained \
  --batch_size 64 \
  --learning_rate 0.001 \
  --split_mode intra \
  --labels "all" \
  --log_dir /path/to/log_dir \

```

</details>

<details>
<summary><b>📜 Option 2: Shell Script</b></summary>

```bash
bash train_models_multi_intra.sh
```

</details>

#### 📋 Parameter Reference

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--dataset_root` | Root directory of the dataset | Path |
| `--data` | Data modality | `rgb`, `depth`, `ir`, `thermal` |
| `--epochs` | Number of training epochs | Integer (default: 15) |
| `--gpu` | GPU device number | Integer |
| `--network` | Network architecture | `resnet18`, `resnet34`, `resnet50`, `vit_b_16` |
| `--weights` | Weight initialization | `pretrained`, `scratch` |
| `--batch_size` | Batch size for training | Integer (default: 64) |
| `--learning_rate` | Learning rate | Float (default: 0.001) |
| `--split_mode` | Data splitting mode | `cross_subject`, `intra` |
| `--oversample` | Enable minority class oversampling | Flag |
| `--labels` | Label frequency rank range | String (e.g., "10,30") or "all" |
| `--cross_user_id` | Test user ID in cross_user mode | Integer |

---

### 🦴 Skeleton Cross_trail Training

```bash
cd skeleton

CUDA_VISIBLE_DEVICES=4,6 python train.py \
  --train_dir cross_trial_train.txt \
  --test_dir cross_trial_test.txt \
  --config ./configs/dstformer.yaml
```

> 📖 See `skeleton/readme.md` for detailed configuration.

---

### 📡 Radar Cross_trail Training

```bash
cd radar

bash ./train_radar_mix.sh
```


---

### 📱 IMU Cross_trail Training

```bash
cd imu

bash ./command_accgyrmag_transformer_crosstrail.sh
```

> 📖 See `imu/readme.md` for detailed configuration.

---

### cross-trail-remove long tail experiments

#### rgb
```bash
cd rgb 

python train_models_cross_multi.py \
  --dataset_root /path/to/dataset \
  --data rgb \
  --epochs 15 \
  --gpu 0 \
  --network resnet50 \
  --weights pretrained \
  --batch_size 64 \
  --learning_rate 0.001 \
  --split_mode intra \
  --oversample \
  --labels "10,30" \
  --log_dir /path/to/log_dir \
```

#### skeleton
```bash
cd skeleton

CUDA_VISIBLE_DEVICES=1,3 python train.py --train_dir cross_subject_train_top20_test1.txt --test_dir cross_subject_test_top20_test1.txt --config ./configs/dstformer.yaml
```

#### imu
```bash
cd imu

bash command_activity20_accgyrmag_resampling_crossuser.sh
```


### 🎯 Cross-Subject Training (RGB)

```bash
cd rgb
cd cross_subject
```

| Script | Description |
|--------|-------------|
| `train_supervised_44.sh` | Fast baseline cross-subject training |
| `train_supervised_lt.sh` | Resampled cross-subject training |
| `train_contra_all_users_44.sh` | Contrastive learning (all actions) |
| `train_10_users_contra.sh` | Contrastive learning (resampled actions) |
| `train_10_users_contra_remove_env.sh` | Contrastive learning (without env variation) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📧 Contact

<div align="center">

**For questions or suggestions, please contact:**

📮 **Email**: [syjiang@ie.cuhk.edu.hk](mailto:syjiang@ie.cuhk.edu.hk)

🏫 **The Chinese University of Hong Kong**

</div>

---

<div align="center">

⭐ **If you find this project helpful, please consider giving it a star!** ⭐

</div>
