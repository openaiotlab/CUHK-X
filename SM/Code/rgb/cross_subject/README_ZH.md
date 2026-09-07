# CUHK-X RGB Cross-Subject Action Recognition Training

## 目录概述

本目录包含基于RGB模态的跨用户(cross-subject)动作识别训练脚本,支持多种训练策略:
- **监督学习(Supervised Learning)**: 纯监督分类训练
- **对比学习(Contrastive Learning)**: SimCLR框架的自监督+监督混合训练
- **环境感知训练(Environment-Aware)**: 考虑环境差异的数据划分策略（由于30个user是在两个不同的环境中完成的数据统计，user1-15为第一个环境，user16-30为第二个环境）

## 目录结构

```
cross_subject/
├── 训练脚本 (Python)
│   ├── train_supervised_44.py          # 全局44类监督学习
│   ├── train_supervised_lt.py           # 指定动作子集监督学习
│   ├── simclr_44.py                    # 44类SimCLR对比学习
│   ├── simclr_10.py                    # 10类SimCLR(跨环境)
│   ├── simclr_10_remove_env.py         # 10类SimCLR(环境感知)
│   └── train_models_cross_multi.py     # 通用多模态训练框架
│
├── 批处理脚本 (Bash)
│   ├── train_supervised_44.sh          # 批量运行44类监督训练
│   ├── train_supervised_a.sh           # 批量运行指定动作监督训练
│   ├── train_contra_all_users_44.sh    # 批量运行44类对比学习
│   ├── train_10_users_contra.sh        # 批量运行10类对比学习(跨环境)
│   ├── train_10_users_contra_remove_env.sh  # 批量运行10类对比学习(环境感知)
│   └── train_models_multi_cross.sh     # 通用多模态训练启动脚本
│
└── 日志目录 (自动生成)
    ├── logs_supervised_all_subject/         # 44类监督训练日志
    ├── supervised_logs_lt/                  # 指定动作监督训练日志
    ├── contra_allusers_logs/                # 44类对比学习日志
    ├── training_logs_contra_lt/             # 10类对比学习日志(跨环境即两个环境)
    └── training_logs_contra_lt_remove_env/  # 10类对比学习日志(环境感知)
```

## 训练方法详解

### 1. 监督学习 (Supervised Learning)

纯监督分类训练,使用ResNet18作为骨干网络。

#### 1.1 全局44类监督学习

**脚本**: `train_supervised_44.py`  
**批处理**: `train_supervised_44.sh`

**训练策略**:
- **训练集**: 其他29个用户的所有44类动作数据
- **测试集**: 指定测试用户的所有44类动作数据
- **网络**: ResNet18 (ImageNet预训练)
- **优化器**: Adam (lr=1e-4)
- **数据增强**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter

**使用方法**:

```bash
# 单用户训练
python train_supervised_44.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --image_size 224

# 批量训练多用户
bash train_supervised_44.sh
```

**配置修改**:
- 编辑`train_supervised_44.sh`中的`USERS`变量选择测试用户
- 默认测试用户: `user1, user7, user13, user20, user26`
- GPU设置: 修改脚本中的`export CUDA_VISIBLE_DEVICES`

**输出**:
- 日志目录: `logs_supervised_all_subject/`
- 评估指标: Accuracy, Precision, Recall, F1-Score

---

#### 1.2 指定动作子集监督学习

**脚本**: `train_supervised_lt.py`  
**批处理**: `train_supervised_lt.sh`

**训练策略**:
- **训练集**: 其他29个用户的指定K类动作数据
- **测试集**: 指定测试用户的相同K类动作数据
- **类别映射**: 使用局部标签 (0 到 K-1)
- **默认动作**: 6, 7, 9, 11, 12, 20, 21, 32, 36, 37 (10类常见动作)

**使用方法**:

```bash
# 单用户训练
python train_supervised_lt.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --image_size 224 \
    --supervised_action_ids 6 7 9 11 12 20 21 32 36 37

# 批量训练
bash train_supervised_lt.sh
```

**关键参数**:
- `--supervised_action_ids`: 指定动作ID列表(空格分隔)
- `--exact_match`: 使用精确匹配模式(推荐)

**输出**:
- 日志目录: `supervised_logs_lt/`

---

### 2. 对比学习 (SimCLR Contrastive Learning)

结合自监督对比学习和监督分类的混合训练策略。

#### 2.1 全局44类对比学习

**脚本**: `simclr_44.py`  
**批处理**: `train_contra_all_users_44.sh`

**训练策略**:
- **监督分支**: 其他29个用户 × 44类动作 (有标签)
- **对比分支**: 测试用户所有数据 (无标签,SimCLR对比学习)
- **损失函数**: `Loss = Loss_supervised + Loss_contrastive`
- **网络结构**: ResNet18 + Projection Head (512→128)

**使用方法**:

```bash
# 单用户训练
python simclr_44.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --temperature 0.5 \
    --image_size 224

# 批量训练
bash train_contra_all_users_44.sh
```

**关键参数**:
- `--temperature`: NT-Xent损失温度参数 (默认0.5)
- `--num_users`: 总用户数 (默认30)

**对比学习数据增强**:
```python
RandomResizedCrop(scale=(0.5, 1.0))
RandomHorizontalFlip()
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)
RandomGrayscale(p=0.2)
GaussianBlur(kernel_size, sigma=(0.1, 2.0))
```

**输出**:
- 日志目录: `contra_allusers_logs/`

---

#### 2.2 指定动作对比学习 (跨环境版本)

**脚本**: `simclr_10.py`  
**批处理**: `train_10_users_contra.sh`

**训练策略**:
- **监督分支**: 其他29个用户(跨环境) × 指定K类动作
- **对比分支**: 测试用户所有数据
- **测试集**: 测试用户的指定K类动作数据
- **环境**: 不区分Environment A (user1-15) 和 Environment B (user16-30)

**使用方法**:

```bash
# 单用户训练
python simclr_10.py \
    --test_user user1 \
    --dataset_path /path/to/RGB \
    --batch_size 64 \
    --epochs 20 \
    --lr 1e-4 \
    --temperature 0.5 \
    --image_size 112 \
    --supervised_action_ids 6 7 9 11 12 20 21 32 36 37

# 批量训练
bash train_10_users_contra.sh
```

**数据规模**:
- 训练数据: ~116,000 样本 (29用户 × 10动作 × ~400样本/动作)
- 对比数据: ~4,400 样本 (1用户 × 10动作 × ~400样本/动作)

**输出**:
- 日志目录: `training_logs_contra_lt/`

---

#### 2.3 指定动作对比学习 (环境感知版本)

**脚本**: `simclr_10_remove_env.py`  
**批处理**: `train_10_users_contra_remove_env.sh`

**训练策略** (核心改进):
- **环境感知划分**: 
  - Environment A: user1-15
  - Environment B: user16-30
- **训练数据**: 仅使用同环境的14个用户数据
- **动作交集计算**: 训练动作 = 指定动作 ∩ 测试用户实际执行动作
- **损失权重**: `Loss = 1.0 × Loss_supervised + 0.5 × Loss_contrastive`

**使用方法**:

```bash
# 单用户训练
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

# 批量训练
bash train_10_users_contra_remove_env.sh
```

**关键特性**:
- `get_environment_users()`: 自动识别用户所属环境
- `get_user_executed_actions()`: 检测用户实际执行的动作
- 动作交集验证: 确保测试集不包含空类别

**数据规模对比**:
| 版本 | 训练用户数 | 样本数 | 环境隔离 |
|------|----------|--------|---------|
| simclr_10.py | 29 (跨环境) | ~116K | ❌ |
| simclr_10_remove_env.py | 14 (同环境) | ~56K | ✅ |

**输出**:
- 日志目录: `training_logs_contra_lt_remove_env/`

---

### 3. 通用多模态训练框架

**脚本**: `train_models_cross_multi.py`  
**批处理**: `train_models_multi_cross.sh`

**训练策略**:
- 支持多种数据模态: RGB, Depth, IR, Thermal
- 支持多种网络架构: ResNet18/34/50, ViT-B/16
- 支持cross-subject和intra-split两种划分模式
- 自动类别不平衡处理(可选oversampling)

**使用方法**:

```bash
# 编辑配置
vim train_models_multi_cross.sh
# 修改以下参数:
DATASET_ROOT="/path/to/data"
DATA="rgb"              # rgb, depth, ir, thermal
NETWORK="resnet50"      # resnet18, resnet34, resnet50, vit_b_16
SPLIT_MODE="cross"      # cross, intra
CROSS_USER_ID=5         # 测试用户ID (仅cross模式)
LABELS="all"            # "all" 或 "10,30" (频率排名范围)
OVERSAMPLE_FLAG=""      # 设置为"--oversample"启用过采样

# 运行训练
bash train_models_multi_cross.sh
```

**关键参数**:
- `--split_mode cross`: 留一用户测试 (LOSO)
- `--split_mode intra`: 80/20随机划分
- `--labels "10,30"`: 使用频率排名10-30的动作
- `--oversample`: 启用少数类过采样
- `--weights pretrained`: 使用ImageNet预训练权重

**输出**:
- 日志目录: `logs_baseline/{data}_{split_mode}_user{id}_{timestamp}/`
- 自动生成训练摘要: `experiment_summary.txt`

---

## 快速启动指南

### 步骤1: 环境配置

```bash
# 进入工作目录
cd ./rgb/cross_subject

# 安装依赖
pip -r install requirements.txt
```

### 步骤2: 数据集准备

确保数据集结构如下:

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

### 步骤3: 选择训练策略

根据实验需求选择对应的脚本:

```bash
# 1. 快速基线测试 (44类监督学习)
bash train_supervised_44.sh

# 2. 少样本学习场景 (10类监督学习)
bash train_supervised_a.sh

# 3. 半监督学习场景 (44类对比学习)
bash train_contra_all_users_44.sh

# 4. 跨环境泛化 (10类对比学习-跨环境)
bash train_10_users_contra.sh

# 5. 同环境优化 (10类对比学习-环境感知)
bash train_10_users_contra_remove_env.sh

# 6. 自定义实验 (通用框架)
bash train_models_multi_cross.sh
```

### 步骤4: 修改测试用户列表

在bash脚本中修改`USERS`数组:

```bash
# 默认配置 (覆盖两个环境)
USERS=("user1" "user7" "user13" "user20" "user26")

# 测试Environment A所有用户 (user1-15)
USERS=($(seq -f "user%g" 1 15))

# 测试Environment B所有用户 (user16-30)
USERS=($(seq -f "user%g" 16 30))

# 测试所有30个用户
USERS=($(seq -f "user%g" 1 30))
```

### 步骤5: GPU配置

修改GPU设备:

```bash
# 在bash脚本中
export CUDA_VISIBLE_DEVICES=2

# 或在命令行临时设置
CUDA_VISIBLE_DEVICES=0,1 bash train_supervised_44.sh
```

---

## 训练脚本详细参数

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_user` | str | 必需 | 测试用户ID (如user1) |
| `--dataset_path` | str | 必需 | RGB数据集根目录 |
| `--batch_size` | int | 64 | 批次大小 |
| `--epochs` | int | 20 | 训练轮数 |
| `--lr` | float | 1e-4 | 学习率 |
| `--image_size` | int | 224 | 图像尺寸 |

### SimCLR特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--temperature` | float | 0.5 | NT-Xent损失温度参数 |
| `--supervised_action_ids` | int+ | 必需 | 监督学习动作ID列表 |
| `--exact_match` | flag | False | 使用精确匹配模式 |
| `--num_users` | int | 30 | 总用户数 |

### 多模态框架参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | rgb | 数据模态 (rgb/depth/ir/thermal) |
| `--network` | str | resnet50 | 网络架构 |
| `--weights` | str | pretrained | 权重初始化 (pretrained/scratch) |
| `--split_mode` | str | cross | 数据划分模式 (cross/intra) |
| `--cross_user_id` | int | 必需 | 测试用户ID (仅cross模式) |
| `--labels` | str | all | 动作频率范围 (如"10,30") |
| `--oversample` | flag | False | 启用少数类过采样 |

---

## 实验结果分析

### 日志文件格式

每个训练脚本会生成时间戳日志:

```
{log_dir}/{test_user}_{timestamp}.log
```

日志内容包括:
```
Epoch [1/20] Train Loss: 2.3456 Train Acc: 45.67%
Epoch [1/20] Test Loss: 2.1234 Test Acc: 52.34%
Precision: 0.5123, Recall: 0.5234, F1: 0.5178
```

### 模型保存位置

```
models/
├── user1_supervised_44.pth          # 44类监督模型
├── user1_supervised_10.pth          # 10类监督模型
├── user1_simclr_44.pth              # 44类对比学习模型
├── user1_simclr_10.pth              # 10类对比学习模型(跨环境)
└── user1_simclr_env_aware.pth       # 10类对比学习模型(环境感知)
```

### 批量结果统计

使用以下命令提取所有用户的最终准确率:

```bash
# 提取监督学习结果
grep "Test Acc:" logs_supervised_all_subject/*.log | tail -n 5

# 提取对比学习结果
grep "Test Acc:" training_logs_contra_lt/*.log | tail -n 5

# 生成CSV统计报告
for log in logs_supervised_all_subject/*.log; do
    user=$(basename $log | cut -d'_' -f1)
    acc=$(grep "Test Acc:" $log | tail -1 | grep -oP '\d+\.\d+%')
    echo "$user,$acc"
done > supervised_results.csv
```

---

## 训练方法对比

| 方法 | 训练数据 | 测试数据 | 优势 | 适用场景 |
|------|---------|---------|------|---------|
| **train_supervised_44** | 29用户×44类 | 1用户×44类 | 简单直接,基线强 | 全动作识别 |
| **train_supervised_a** | 29用户×K类 | 1用户×K类 | 专注少数类 | 特定动作识别 |
| **simclr_44** | 29用户×44类(标注)<br>+1用户×44类(无标注) | 1用户×44类 | 利用目标域数据 | 半监督学习 |
| **simclr_10** | 29用户×K类(标注)<br>+1用户×全部(无标注) | 1用户×K类 | 对比学习增强 | 跨环境泛化 |
| **simclr_10_remove_env** | 14用户×K类(标注)<br>+1用户×全部(无标注) | 1用户×K类 | 环境隔离,减少分布差异 | 同环境优化 |

---

## 常见问题

### Q1: 如何选择合适的训练方法?

- **全动作识别**: 使用`train_supervised_44.sh`或`simclr_44.sh`
- **少样本/特定动作**: 使用`train_supervised_a.sh`
- **目标域有无标注数据**: 使用`simclr_10.sh`或`simclr_10_remove_env.sh`
- **跨环境泛化**: 使用`simclr_10.sh` (29用户)
- **同环境优化**: 使用`simclr_10_remove_env.sh` (14用户)

### Q2: 为什么环境感知版本使用更少的训练数据?

Environment A (user1-15)和Environment B (user16-30)存在光照、背景等差异。跨环境训练虽然数据多,但引入分布偏移。环境感知版本通过隔离环境:
- 减少52%训练样本 (116K → 56K)
- 消除环境导致的分布差异
- 提升同环境泛化能力

### Q3: ColorJitter的hue参数为什么设置为0?

旧版torchvision存在bug,`hue=0.1`会导致`Python integer -XX out of bounds for uint8`错误。所有脚本已修复为`hue=0`。

### Q4: 如何修改对比学习的损失权重?

编辑Python脚本中的损失计算部分:

```python
# simclr_10.py (原始版本)
loss = supervised_loss + contrastive_loss  # 权重1:1

# simclr_10_remove_env.py (环境感知版本)
loss = supervised_loss + 0.5 * contrastive_loss  # 权重1:0.5
```

### Q5: 如何添加新的数据增强?

编辑`get_train_transform()`函数:

```python
def get_train_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        
        # 添加新的增强
        transforms.RandomApply([
            transforms.RandomRotation(degrees=15)
        ], p=0.3),
        
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0)
        ], p=0.8),
        # ... 其他增强
    ])
```

### Q6: 如何查看训练进度?

```bash
# 实时查看日志
tail -f training_logs_contra_lt/user1_*.log

# 监控GPU使用
watch -n 1 nvidia-smi

# 提取准确率曲线
grep "Test Acc:" training_logs_contra_lt/user1_*.log
```

### Q7: 训练时显存不足怎么办?

```bash
# 方法1: 减小batch_size
--batch_size 32  # 默认64

# 方法2: 减小图像尺寸
--image_size 112  # 默认224

# 方法3: 使用更小的网络
# 编辑Python脚本,将ResNet18改为更轻量级网络

# 方法4: 使用梯度累积 (需修改代码)
```

### Q8: 如何导出实验结果表格?

```bash
# 创建结果统计脚本
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

## 高级用法

### 自定义动作子集

```bash
# RGB常用10类动作 (默认)
SUP_ACTIONS="6 7 9 11 12 20 21 32 36 37"

# Depth常用10类动作
SUP_ACTIONS="2 6 7 11 20 21 32 34 36 37"

# 自定义5类核心动作
SUP_ACTIONS="6 7 9 11 12"
```

### 并行训练多用户

```bash
# 使用GNU parallel (需安装)
parallel -j 4 python train_supervised_44.py --test_user {} ::: user{1..30}

# 或使用后台任务
for user in user{1..5}; do
    CUDA_VISIBLE_DEVICES=$((i%4)) python train_supervised_44.py \
        --test_user $user &
    ((i++))
done
wait
```

### 自动超参数搜索

```bash
# 网格搜索学习率和batch_size
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

## 代码架构说明

### 模型结构

所有脚本使用统一的模型架构:

```python
class UnifiedModel(nn.Module):
    def __init__(self, num_classes, projection_dim=128):
        super().__init__()
        # 骨干网络
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # 移除最后的FC层
        
        # 分类头 (监督学习)
        self.classifier = nn.Linear(512, num_classes)
        
        # 投影头 (对比学习)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
```

### 损失函数

```python
# 监督损失
supervised_loss = nn.CrossEntropyLoss()(logits, labels)

# 对比损失 (NT-Xent)
def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.mm(z, z.T) / temperature
    # ... 详见代码实现
    return loss
```

### 数据加载流程

```
1. get_dataloaders()
   ├─> 扫描数据集目录
   ├─> 构建监督数据列表 (其他用户)
   ├─> 构建对比数据列表 (测试用户)
   └─> 构建测试数据列表 (测试用户)

2. ActionDataset
   ├─> 监督模式: 返回 (image, label)
   └─> 对比模式: 返回 (view1, view2)

3. DataLoader
   ├─> 批次采样
   ├─> 数据增强 (on-the-fly)
   └─> 返回batch
```

---

## 引用与参考

如使用本代码,请引用:

```bibtex
@article{cuhkx2025,
  title={CUHK-X: Multi-modal Cross-subject Action Recognition Dataset},
  author={Your Name},
  journal={Conference/Journal},
  year={2025}
}
```

**相关方法**:
- SimCLR: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## 联系方式

如有问题或建议,请联系:
- Email: syjiang@ie.cuhk.edu.hk
- GitHub Issues: [项目仓库链接]

---

## 更新日志

- **2025-12-02**: 创建完整README文档
- **2025-11-XX**: 修复ColorJitter hue参数溢出bug
- **2025-11-XX**: 添加环境感知训练版本 (simclr_10_remove_env.py)
- **2025-10-XX**: 初始版本发布
