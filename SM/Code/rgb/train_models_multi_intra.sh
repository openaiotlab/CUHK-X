#!/bin/bash
# train_script.sh - Automated training script for image classification models
# Usage: ./train_script.sh [experiment_name] [gpu_id]

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ======================
# Configuration Section
# ======================
DATASET_ROOT="/YOUR/PATH/TO/CUHKX/SM/Data"
DATA="rgb" #"rgb", "depth", "ir", "thermal"
LOG_ROOT="./logs_all"
SCRIPT_PATH="./train_models_cross_multi.py"

# Default parameters (can be overridden)
GPU_ID=4  # Physical GPU ID (0-6), will automatically set CUDA_VISIBLE_DEVICES
EPOCHS=15
BATCH_SIZE=64
LEARNING_RATE=0.001
NETWORK="resnet50"
WEIGHTS="pretrained"
SPLIT_MODE="intra"  
OVERSAMPLE_FLAG=" "  # Empty means disabled, set to "--oversample" to enable
LABELS="all"        # Use "all" for all labels or "10,30" for range

# =====================
# Environment Setup
# =====================
# Set CUDA_VISIBLE_DEVICES based on GPU_ID
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Create log directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR="${LOG_ROOT}/${DATA}_${SPLIT_MODE}_${TIMESTAMP}"

mkdir -p "${LOG_DIR}"

# ====================
# Parameter Validation
# ====================
# Validate GPU availability
if ! nvidia-smi -i "${GPU_ID}" > /dev/null 2>&1; then
    echo "ERROR: GPU ${GPU_ID} not available. Available GPUs:"
    nvidia-smi --query-gpu=index,name --format=csv
    exit 1
fi

# Validate dataset path
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: Dataset root directory not found at ${DATASET_ROOT}"
    exit 1
fi

# ===================
# Training Execution
# ===================
echo "==========================================="
echo "Starting training experiment"
echo "Physical GPU: ${GPU_ID} (mapped to cuda:0)"
echo "Log directory: ${LOG_DIR}"
echo "Split mode: ${SPLIT_MODE}"
echo "==========================================="

# Build training command with unbuffered output

# Run the training directly (no eval to avoid subshell issues)
# Always use --gpu 0 because CUDA_VISIBLE_DEVICES makes the selected GPU appear as cuda:0
python -u "${SCRIPT_PATH}" \
    --dataset_root "${DATASET_ROOT}" \
    --data "${DATA}" \
    --epochs ${EPOCHS} \
    --gpu 0 \
    --network "${NETWORK}" \
    --weights "${WEIGHTS}" \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --split_mode "${SPLIT_MODE}" \
    ${OVERSAMPLE_FLAG} \
    --labels "${LABELS}" \
    --log_dir "${LOG_DIR}"

# =====================
# Post-Processing
# =====================
echo ""
echo "==========================================="
echo "Training completed!"
echo "Results saved to: ${LOG_DIR}"
echo "==========================================="

# Generate summary report
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"
cat > "${SUMMARY_FILE}" <<EOF
Experiment Summary
===================
Timestamp: ${TIMESTAMP}
GPU Used: ${GPU_ID}
Model: ${NETWORK} (${WEIGHTS})
Dataset: ${DATASET_ROOT}
Data Modality: ${DATA}
Split Mode: ${SPLIT_MODE}
Cross User ID: ${CROSS_USER_ID}
Oversampling: ${OVERSAMPLE_FLAG:-Disabled}
Labels Used: ${LABELS}
Final Logs: ${LOG_DIR}
EOF

echo "Summary report created at ${SUMMARY_FILE}"