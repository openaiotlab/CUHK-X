# CUHK-X-VLM: Multimodal Video-Language Understanding Toolkit

CUHK-X-VLM is a toolkit for multimodal video tasks, supporting action selection, video captioning, emotion analysis, 
and other tasks, with automatic evaluation (e.g., BLEU/METEOR). It is compatible with mainstream multimodal models 
(InternVL/QwenVL/Video-LLaVA/Video-Chat, etc.).

# Project Directory Structure
```
LM
├──CUHK-X-VLM/
│   ├── src/
│      ├── action_selection/                        # Task of selecting actions from 40 categories
│      │    ├── predictions
│      │    ├── scores
│      ├── context_analysis/                        # Task of emotion analysis
│      ├── sequential_action_recording/             # Task of action sequencing
│      └── task_caption/                            # Task of generating video captions
│      └── task_HARn/                               # Task of action prediction
├── GT_folder   # Ground truth label data
├── LM_data     # Raw collected data in image format
├── LM_video    # Processed video data
├── Models      # Saved models
├── requirements.txt
├── README_ZH.md
└── README.md
```

# Usage

## 1. Create and activate Conda environment
```
# clone the repository
git clone git@github.com:siyang-jiang/CUHK-X.git
cd CUHK-X/LM

# create virtual environment
conda create -n cuhkx python==3.9
conda activate cuhkx
```

## 2. Install dependencies
```
# Dependencies based on CUDA: 11.8
pip install -r requirements.txt
```

## 3. Download Dataset and Models

### 3.1 Download Dataset
You can download the datasets (GT_folder, LM_video, LM_data) from the homepage [CUHK-X](https://siyang-jiang.github.io/CUHK-X/) and save them in the corresponding directories.

### 3.2 Download Models
Models will be saved in the Models folder. Each task can use different models by modifying parameters.
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Models/Qwen2.5-VL-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir Models/Qwen2.5-VL-7B-Instruct
huggingface-cli download OpenGVLab/InternVL2-2B --local-dir Models/InternVL2-2B
huggingface-cli download OpenGVLab/InternVL2-8B --local-dir Models/InternVL2-8B
huggingface-cli download OpenGVLab/VideoChat-R1_7B --local-dir Models/VideoChat-R1_7B
huggingface-cli download LanguageBind/Video-LLaVA-7B-hf --local-dir Models/Video-LLaVA-7B-hf
```

## 4. Run

### 4.1 模型输出

First, add parameters to `exp.sh` For example, to perform the `action_selection` task based on the `depth` modality using the `InternVL-2B` model, add the following script to `exp.sh`:
```
python CUHK-X-VLM/src/action_selection/main_internvl_choices.py --modality depth --model_size 2B
```
For multi-GPU users, you can add the following script `CUDA_VISIBLE_DEVICES=0,1,2,3` to identify multiple GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python CUHK-X-VLM/src/action_selection/main_internvl_choices.py --modality depth --model_size 2B
```

Then, execute the following script, and the generated results will be saved in the `action_selection/predictions` folder.
```
bash CUHK-X-VLM/src/action_selection/exp.sh
```

### 4.2 Calculate Scores

Add the following script to `score.sh`, which includes two parameters: `modality` and `method`.
```
python CUHK-X-VLM/src/action_selection/calculate_score.py --modality depth --method internvl2B
```
Execute `score.sh`, and the generated results will be saved in the `action_selection/scores folder`.
```
bash CUHK-X-VLM/src/action_selection/score.sh
```


More examples can be found in the `exp.sh` file.
