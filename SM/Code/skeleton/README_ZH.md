# CUHK-X Skeleton-based Action Recognition Training

## 概述

本项目使用DSTformer模型进行基于骨骼数据的动作识别训练。支持多种数据划分模式,包括cross-trial(随机80/20划分)、cross-subject(留一用户测试)和resampled_cross-subject-trial。

## 环境要求

确保已安装Python 3.9+和所需依赖库:

```bash
pip install -r requirements.txt
```

主要依赖:
- PyTorch
- sklearn
- tqdm
- numpy
- wandb (用于训练日志记录)

## 使用流程

### 步骤1: 进入skeleton目录

```bash
cd YOUR/PATH/TO/SM/Code/skeleton
```

### 步骤2: 配置动作类别数量

在`configs/`目录下修改配置文件中的`action_classes`参数。

编辑`configs/dstformer.yaml`:

```yaml
# Data
dataset: cuhkx
action_classes: 44  # 根据实际数据集修改类别数量
```

**注意**: 如果使用子集数据(如特定用户或动作),需要相应调整类别数量。

### 步骤3: 生成数据划分文件

修改`split_data.py`中的数据路径:

```python
# 第6-7行,修改为你的数据集路径
root_dir = '/path/to/your/SM_data/Skeleton'  # 骨骼数据根目录
save_dir = '/path/to/split_data_results'      # 划分结果保存目录
```

运行数据划分脚本:

```bash
python split_data.py
```

脚本会生成以下文件到`split_data_results/`目录:

#### Cross-trial模式 (随机80/20划分)
- `cross_trial_train.txt` - 训练集 (80%)
- `cross_trial_test.txt` - 测试集 (20%)
- `cross_trial_distribution.png` - 数据分布可视化

#### Cross-subject模式 (留一用户测试)
- `cross_subject_train_{user_id}.txt` - 训练集 (其他29个用户)
- `cross_subject_test_{user_id}.txt` - 测试集 (指定用户)
- `cross_subject_distribution_{user_id}.png` - 数据分布可视化

#### Cross-subject-resample模式 (去长尾并选定用户)
- `cross_subject_trial_train_{user_range}.txt` - 训练集 （resample后的训练集）
- `cross_subject_trial_test_{user_range}.txt` - 测试集 
- `cross_subject_trial_{user_range}_distribution.png` - 数据分布可视化


### 步骤4: 训练模型

使用生成的划分文件进行训练:

#### Cross-trial模式训练

```bash
CUDA_VISIBLE_DEVICES=4,6 python train.py \
  --train_dir cross_trial_train.txt \
  --test_dir cross_trial_test.txt \
  --config ./configs/dstformer.yaml
```

#### Cross-subject模式训练 (所有30用户)

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py \
  --train_dir cross_subject_trial_train_30users.txt \
  --test_dir cross_subject_trial_test_30users.txt \
  --config ./configs/dstformer.yaml
```

#### Cross-subject-resample模式训练 (20用户子集)

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py \
  --train_dir cross_subject_train_top20_test1.txt \
  --test_dir cross_subject_test_top20_test1.txt \
  --config ./configs/dstformer.yaml
```

#### Cross-subject模式训练 (留一用户,如user10)

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py \
  --train_dir cross_subject_train_10.txt \
  --test_dir cross_subject_test_10.txt \
  --config ./configs/dstformer.yaml
```

### 步骤5: 模型评估

使用`--evaluate`参数评估已训练模型:

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --train_dir cross_subject_train_10.txt \
  --test_dir cross_subject_test_10.txt \
  --config configs/dstformer.yaml \
  --evaluate checkpoint/cuhkx_cross_subject/DSTformer_cross_subject_train_10/best_epoch.bin
```

## 训练参数说明

### 命令行参数

- `--train_dir`: 训练集划分文件名 (位于split_data_results/)
- `--test_dir`: 测试集划分文件名 (位于split_data_results/)
- `--config`: 模型配置文件路径
- `--evaluate`: 评估模式,指定checkpoint路径
- `--resume`: 从checkpoint恢复训练
- `--checkpoint`: checkpoint保存目录

### 配置文件参数 (dstformer.yaml)

```yaml
# 训练参数
epochs: 100              # 训练轮数
batch_size: 64          # 批次大小
lr_backbone: 0.00001    # 骨干网络学习率
lr_head: 0.0001         # 分类头学习率
weight_decay: 0.01      # 权重衰减
lr_decay: 0.95          # 学习率衰减

# 模型参数
backbone: DSTformer     # 骨干网络架构
num_joints: 17          # 骨骼关节点数量
dim_feat: 64            # 特征维度
dim_rep: 64             # 表示维度
depth: 1                # Transformer深度
num_heads: 2            # 注意力头数量
hidden_dim: 256         # 隐藏层维度
dropout_ratio: 0.5      # Dropout比率

# 数据增强
random_move: True              # 随机移动增强
scale_range_train: [1, 3]     # 训练集缩放范围
scale_range_test: [2, 2]      # 测试集缩放范围
```

## 数据集结构

骨骼数据应遵循以下目录结构:

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

每个`.npy`文件包含骨骼关节点坐标 (shape: [num_frames, num_joints, 3])。

## 输出结果

### 训练日志

- 自动保存到`checkpoint/`目录
- 使用wandb记录训练曲线(项目名: cuhkx_cross_trial或cuhkx_cross_subject)
- 保存`latest_epoch.bin`和`best_epoch.bin`

### 评估指标

训练和评估过程会输出以下指标:

- **Accuracy@1/5**: Top-1和Top-5准确率
- **Precision**: 宏平均精确率
- **Recall**: 宏平均召回率
- **F1-Score**: 宏平均F1分数
- **AUC-ROC**: 多分类ROC曲线下面积

针对少数类(minority classes)和多数类(majority classes)会单独计算性能指标。


## 许可证

本项目遵循MIT许可证。

## 联系方式

如有问题或建议,请联系 [syjiang@ie.cuhk.edu.hk]
