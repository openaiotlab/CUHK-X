# CUHK-X-VLM: 多模态视频理解大模型工具

CUHK-X-VLM 是一个用于多模态视频任务的工具包，支持动作选择、视频字幕、情感分析等任务，并自动评估（例如 BLEU/METEOR）。它与主流多模态模型（InternVL/QwenVL/Video-LLaVA/Video-Chat 等）兼容。

# 项目目录
```
LM
├──CUHK-X-VLM/
│   ├── src/
│      ├── action_selection/                        # 从40类动作挑选动作的任务
│      │    ├── predictions
│      │    ├── scores
│      ├── context_analysis/                        # 情感分析的任务
│      ├── sequential_action_recording/             # 动作排序的任务
│      └── task_caption/                            # 生成视频字幕的任务
│      └── task_HARn/                               # 预测动作的任务
├── GT_folder   # 真实标签数据
├── LM_data     # 原始采集的数据，image格式
├── LM_video    # 经过处理得到的video数据
├── Models      # 保存模型
├── requirements.txt
├── README_ZH.md
└── README.md
```

# 用法

## 1. 创建conda环境
```
# clone the repository
git clone git@github.com:siyang-jiang/CUHK-X.git
cd CUHK-X/LM

# creat virtual env
conda create -n cuhkx python==3.9
conda activate cuhkx
```

## 2. 安装依赖
```
# 基于 CUDA: 11.8 的依赖
pip install -r requirements.txt
```

## 3. 下载数据集和模型

### 3.1 下载数据集
你可以在主页 [CUHK-X](https://siyang-jiang.github.io/CUHK-X/)下载数据集(GT_folder, LM_video, LM_data)，并保存在对应的目录下。

### 3.2 下载模型

模型会保存至Models文件夹下。每个任务可以通过修改参数，替换不同的模型执行。
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Models/Qwen2.5-VL-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Models/Qwen2.5-VL-7B-Instruct
huggingface-cli download OpenGVLab/InternVL2-2B --local-dir Models/InternVL2-2B
huggingface-cli download OpenGVLab/InternVL2-8B --local-dir Models/InternVL2-8B
huggingface-cli download OpenGVLab/VideoChat-R1_7B --local-dir Models/VideoChat-R1_7B
huggingface-cli download LanguageBind/Video-LLaVA-7B-hf --local-dir Models/Video-LLaVA-7B-hf
```

## 4. 运行

### 4.1 模型输出

首先，添加`exp.sh`的参数。例如，用`InternVL-2B`模型执行`action_selection`基于`depth`模态的任务，则在`exp.sh`添加如下脚本：
```
python CUHK-X-VLM/src/action_selection/main_internvl_choices.py --modality depth --model_size 2B
```
对于多GPU用户，可以添加如下脚本`CUDA_VISIBLE_DEVICES=0,1,2,3`识别多卡：
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/action_selection/main_internvl_choices.py --modality depth --model_size 2B
```

然后，执行以下脚本，生成结果会保存在`action_selection/predictions`文件夹下。
```
bash CUHK-X-VLM/src/action_selection/exp.sh
```
### 4.2 计算分数

在`score.sh`添加如下脚本，包含`modality`和`method`两个参数。
```
python CUHK-X-VLM/src/action_selection/calculate_score.py --modality depth --method internvl2B
```
执行`score.sh`，生成结果会保存在`action_selection/scores`文件夹下。
```
bash CUHK-X-VLM/src/action_selection/score.sh
```


更多样例可参考`exp.sh`文件。