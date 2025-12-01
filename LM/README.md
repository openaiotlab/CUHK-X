# CUHK-X-VLM: Multimodal Video-Language Understanding Toolkit

CUHK-X-VLM is a toolkit for video-language multimodal tasks, supporting action selection, video captioning, context analysis, and automated evaluation (e.g., BLEU/METEOR). It is compatible with mainstream multimodal models (InternVL/QwenVL/Video-LLaVA/Video-Chat, etc.).

# Project Directory Structure
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

# Usage

## 1. Create and activate Conda environment
```
conda create -n cuhkx python=3.9
conda activate cuhkx
```

## 2. Install core dependencies
```
pip install -r requirements.txt
```

## 3. Download the Dataset and Models

### 3.1 Download Dataset
You can dowanload dataset(GT_folder, LM_video, LM_data) in the homepage [CUHK-X](https://siyang-jiang.github.io/CUHK-X/)

### 3.2 Download Models
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Models/Qwen2.5-VL-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Models/Qwen2.5-VL-7B-Instruct
huggingface-cli download OpenGVLab/InternVL2-2B --local-dir Models/InternVL2-2B
huggingface-cli download OpenGVLab/InternVL2-8B --local-dir Models/InternVL2-8B
huggingface-cli download OpenGVLab/VideoChat-R1_7B --local-dir Models/VideoChat-R1_7B
huggingface-cli download LanguageBind/Video-LLaVA-7B-hf --local-dir Models/Video-LLaVA-7B-hf
```

## 4. Run experiments and Calculate evaluation scores

Firstly, modify model size and modality in exp.sh.

Then, execute the scripts.
```
bash CUHK-X-VLM/src/[task]/exp.sh
bash CUHK-X-VLM/src/[task]/score.sh
```
