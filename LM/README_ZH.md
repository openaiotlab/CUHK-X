# CUHK-X-VLM: 多模态视频理解大模型工具

CUHK-X-VLM 是一个用于多模态视频任务的工具包，支持动作选择、视频字幕、上下文分析和自动评估（例如 BLEU/METEOR）。它与主流多模态模型（InternVL/QwenVL/Video-LLaVA/Video-Chat 等）兼容。

# 项目目录
```
LM
├──CUHK-X-VLM/
│   ├── src/
│      ├── action_selection/
│      │    ├── predictions
│      │    ├── scores
│      ├── context_analysis/
│      ├── sequential_action_recording/
│      └── task_caption/
│      └── task_HARn/
├── GT_folder
├── LM_data
├── LM_video
├── Models
├── requirements.txt
├── README_ZH.md
└── README.md
```

# 用法

## 1. 创建conda环境
```
conda create -n cuhkx python=3.9
conda activate cuhkx
```

## 2. 安装依赖
```
pip install -r requirements.txt
```

## 3. 下载数据集和模型

### 3.1 下载数据集
你可以在主页 [CUHK-X](https://siyang-jiang.github.io/CUHK-X/)下载数据集(GT_folder, LM_video, LM_data)

### 3.2 下载模型
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Models/Qwen2.5-VL-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Models/Qwen2.5-VL-7B-Instruct
huggingface-cli download OpenGVLab/InternVL2-2B --local-dir Models/InternVL2-2B
huggingface-cli download OpenGVLab/InternVL2-8B --local-dir Models/InternVL2-8B
huggingface-cli download OpenGVLab/VideoChat-R1_7B --local-dir Models/VideoChat-R1_7B
huggingface-cli download LanguageBind/Video-LLaVA-7B-hf --local-dir Models/Video-LLaVA-7B-hf
```

## 4. 运行

首先，修改`exp.sh`的参数。

然后，执行以下脚本。
```
bash CUHK-X-VLM/src/[task]/exp.sh
bash CUHK-X-VLM/src/[task]/score.sh
```
