# A Large-Scale Multimodal Dataset and Benchmark for Human Action Recognition, Understanding and Reasoning (CUHK-X)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/your-arxiv-id)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](https://your-dataset-link.com)
[![Website](https://img.shields.io/badge/Project-Page-blue)](cuhkx.html)

> **CUHK-X** is a comprehensive multimodal dataset containing **58,445 samples** across **seven modalities** designed for human activity recognition, understanding, and reasoning. It addresses critical gaps in existing HAR datasets by providing synchronized multimodal sensor data with detailed annotations for complex reasoning tasks.

## 🏆 News

- **[Nov 2025]** 🎉 **CUHK-X wins the Best Presentation Award at ANAI Workshop @ MobiCom 2025!** We are honored to receive this recognition for our work on multimodal human action recognition, understanding, and reasoning.

## 🎯 Key Contributions

- **First Multimodal HAU Dataset**: CUHK-X is the first dataset to integrate understanding and reasoning across multiple modalities for human action analysis
- **Large-Scale & Diverse**: 58,445 samples from 30 participants across diverse environments with 7 synchronized modalities
- **Novel Evaluation Framework**: Three comprehensive benchmarks (HAR, HAU, HARn) with 8 distinct tasks
- **LLM-Empowered Annotation**: Innovative prompt-based scene creation framework for logical and spatio-temporal representation

## 📊 Dataset Overview

### Modalities (7 Total)
- **RGB Video**: Standard color video recordings
- **Infrared (IR)**: Thermal imaging for robustness to lighting conditions  
- **Depth**: 3D spatial information from depth cameras
- **Thermal**: Heat signature analysis
- **IMU**: Inertial Measurement Unit sensor data
- **mmWave Radar**: Privacy-preserving motion detection
- **Skeleton**: 3D pose estimation data

### Statistics
- **Total Samples**: 58,445 annotated action samples
- **Participants**: 30 diverse subjects
- **Environments**: 2 (indoor/outdoor) with varying conditions
- **Actions**: 40+ different action categories
- **Data Types**: Both singular actions and sequential activity sequences

## 🏗️ Dataset Structure

The dataset is organized into two main components:

### Small Model Data
- **Focus**: Singular, well-defined actions (similar to traditional datasets)
- **Actions**: 40+ different action categories
- **Samples**: 30,000+ individual action instances
- **Purpose**: Traditional HAR evaluation and baseline comparison

### Large Model Data  
- **Focus**: Sequential actions performed consecutively
- **Purpose**: Temporal and emotional analysis, complex reasoning tasks
- **Features**: Multi-step activity sequences with logical flow
- **Applications**: Human Action Understanding (HAU) and Next Action Reasoning (HARn)

## 🎯 Benchmarks & Tasks

### 1. Human Action Recognition (HAR)
**Objective**: Traditional action classification across modalities
- **Cross-subject evaluation** with Leave-One-Subject-Out (LOSO) protocol
- **Cross-domain performance** analysis 
- **Long-tail distribution** handling
- **Multimodal fusion** strategies

### 2. Human Action Understanding (HAU)  
**Objective**: Comprehend actions through perceptual and contextual integration

**Sub-tasks**:
1. **Action Captioning**: Generate natural language descriptions
2. **Emotion Analysis**: Identify emotional states during activities  
3. **Sequential Action Reordering**: Organize actions chronologically
4. **Action Selection**: Choose relevant actions from candidates

### 3. Human Action Reasoning (HARn)
**Objective**: Infer intentions and causal relationships in action sequences
- **Next Action Prediction**: Predict likely subsequent actions
- **Temporal Reasoning**: Understand action progression logic
- **Contextual Inference**: Consider environmental and situational factors

## 🔬 Technical Highlights

### Novel ActScene Framework
- **LLM-Generated Scenarios**: Consistent and logical activity descriptions
- **Human-in-the-Loop Validation**: Quality assurance for generated content
- **Synchronized Collection**: All modalities captured simultaneously
- **Environmental Diversity**: Multiple settings and conditions

### Hardware Setup
- **Vzense NYX 650**: RGB-D camera for color and depth
- **Texas Instruments Radar**: mmWave sensing for privacy-preserving detection
- **IMU Sensors**: Motion and orientation tracking
- **Thermal Cameras**: Heat signature analysis
- **Synchronized Recording**: Temporal alignment across all modalities

## 📈 Key Findings

### Model Performance Insights
- **Larger models** (7B parameters) consistently outperform smaller ones across tasks
- **QwenVL-7B** and **VLLaVA-7B** demonstrate superior performance in most benchmarks
- **Depth and IR modalities** often provide richer information than RGB for reasoning tasks
- **Cross-subject performance** drops significantly (56.56% vs higher in-domain accuracy)

### Challenging Aspects
- **Domain Shift**: Cross-domain evaluation reveals substantial performance gaps
- **Long-tail Distribution**: Realistic but challenging class imbalance
- **Sequential Reasoning**: Complex temporal understanding requires advanced models
- **Multimodal Fusion**: Optimal combination strategies vary by task

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- PyTorch 1.7+
- CUDA 10.0+ (for GPU acceleration)
- Required packages: see `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/CUHK-X.git
cd CUHK-X

# Install dependencies
pip install -r requirements.txt
```

### Data Access
1. **Request Access**: Contact authors for dataset download link
2. **Data Structure**: Follow the organized directory structure
3. **Loading Example**:
```python
from cuhkx import CUHKXDataset

# Load dataset
dataset = CUHKXDataset(
    data_root='path/to/data',
    modalities=['rgb', 'depth', 'imu'],
    task='har'  # or 'hau', 'harn'
)

# Iterate through samples
for sample in dataset:
    rgb_data = sample['rgb']
    depth_data = sample['depth'] 
    imu_data = sample['imu']
    label = sample['label']
    caption = sample['caption']
```

### Evaluation
```bash
# Run HAR benchmark
python evaluate_har.py --modalities rgb depth imu --model your_model

# Run HAU benchmark  
python evaluate_hau.py --task captioning --model qwen-vl-7b

# Run HARn benchmark
python evaluate_harn.py --model llava-7b
```

## 📋 Benchmark Results

### HAR Performance (Cross-subject LOSO)
| Modality | Baseline | w/ Contrastive | w/o Cross-domain |
|----------|----------|----------------|------------------|
| RGB      | 45.2%    | 52.8%         | 56.56%          |
| IMU      | 38.7%    | 44.3%         | 48.9%           |
| Skeleton | 41.5%    | 47.2%         | 51.3%           |

### HAU Performance (Selected Tasks)
| Model       | Captioning (BLEU-1) | Emotion Analysis | Sequential Reordering |
|-------------|---------------------|------------------|-----------------------|
| QwenVL-7B   | 55.97%             | 77.77%          | 68.5%                |
| VLLaVA-7B   | 22.32%             | 74.2%           | 66.8%                |
| InternVL-8B | 0.59%              | 35.35%          | 45.3%                |

## 🎯 Applications

### Healthcare & Monitoring
- **Cognitive Decline Detection**: Identify forgetfulness or repetitive behaviors
- **Daily Activity Assessment**: Monitor activities of daily living (ADL)
- **Rehabilitation Progress**: Track recovery through activity analysis

### Smart Environments  
- **Home Automation**: Context-aware system responses
- **Security & Safety**: Anomaly detection in activity patterns
- **Human-Computer Interaction**: Natural interface design

### Research & Education
- **Multimodal Learning**: Sensor fusion algorithm development
- **Temporal Reasoning**: Sequential action understanding
- **Privacy-Preserving AI**: Non-visual sensing research

## 🏆 Broader Impact

CUHK-X aims to advance research in:
- **Conventional HAR**: Multimodal algorithms and cross-domain methods
- **LLM Evaluation**: Benchmark for action understanding capabilities  
- **Educational Resource**: Standard dataset for teaching sensor fusion and multimodal reasoning
- **Real-world Deployment**: Bridge the gap between lab and practical applications

## 📝 Citation

If you use CUHK-X in your research, please cite our paper:

<!-- ```bibtex
@inproceedings{jiang2025cuhkx,
  title={CUHK-X: A Large-Scale Multimodal Dataset and Benchmark for Human Action Recognition, Understanding and Reasoning},
  author={Jiang, Siyang and others},
  booktitle={Proceedings of the 26th International Conference on Sensing, Communication, and Networking (SenSys)},
  year={2025}
}
``` -->

## 👥 Contact

For dataset access, questions, or collaborations:
- **Email**: syjiang [AT] ie.cuhk.edu.hk
- **Project Page**: [CUHK-X](https://siyang-jiang.github.io/CUHK-X/)

## 🔗 Related Work

- **PGADA**: [Perturbation-Guided Adversarial Alignment](https://github.com/siyang-jiang/PGADA)
- **ArtFL**: [Federated Learning with Multi-Scale Training](https://github.com/siyang-jiang/ArtFL)

## 📄 License

This dataset is released under the [Creative Commons Attribution 4.0 International License](LICENSE).

---

**Note**: This dataset is designed for research and educational purposes. Please ensure compliance with your institution's ethics guidelines when using human activity data.
