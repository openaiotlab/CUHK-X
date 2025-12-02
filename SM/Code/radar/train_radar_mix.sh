#!/bin/bash
# train_radar_mix.sh - Automated training script for Radar point cloud action recognition
# Usage: ./train_radar_mix.sh [gpu_id]

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ======================
# Configuration Section
# ======================
SCRIPT_PATH="./train_models_radar_mix.py"
LOG_ROOT="./logs_radar_all"

# Default parameters (can be overridden via command line or here)
GPU_ID=${1:-6}          # Default GPU ID, can be passed as first argument
WINDOW_SIZE=20          # Time window length (frames)
MAX_POINTS=112          # Max points per frame
BATCH_SIZE=32           # Batch size
NUM_EPOCHS=40           # Number of training epochs
LEARNING_RATE=1e-3      # Learning rate
NUM_WORKERS=8           # Data loading workers

# Dataset path (single or multiple)

DATA_ROOT="/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-example/SM/Data"


# =====================
# Environment Setup
# =====================
# Set CUDA_VISIBLE_DEVICES based on GPU_ID
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Create log directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${LOG_ROOT}/radar_mix_${TIMESTAMP}"
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
if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: Dataset root directory not found at ${DATA_ROOT}"
    exit 1
fi

# Validate script exists
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Training script not found at ${SCRIPT_PATH}"
    exit 1
fi

# ===================
# Training Execution
# ===================
echo "==========================================="
echo "     Radar Point Cloud Action Recognition"
echo "==========================================="
echo "Start Time: $(date)"
echo "Physical GPU: ${GPU_ID} (mapped to cuda:0)"
echo "Log directory: ${LOG_DIR}"
echo "Dataset: ${DATA_ROOT}"
echo ""
echo "Hyperparameters:"
echo "  Window Size: ${WINDOW_SIZE}"
echo "  Max Points: ${MAX_POINTS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Num Workers: ${NUM_WORKERS}"
echo "==========================================="

# Create a temporary Python config override script
CONFIG_OVERRIDE=$(cat <<EOF
import sys
sys.path.insert(0, '.')
import train_models_radar_mix as train_script

# Override config
train_script.config["device"] = "cuda:0"
train_script.config["window_size"] = ${WINDOW_SIZE}
train_script.config["max_points"] = ${MAX_POINTS}
train_script.config["batch_size"] = ${BATCH_SIZE}
train_script.config["num_epochs"] = ${NUM_EPOCHS}
train_script.config["learning_rate"] = ${LEARNING_RATE}
train_script.config["num_workers"] = ${NUM_WORKERS}
train_script.config["data_root"] = "${DATA_ROOT}"
train_script.config["log_dir"] = "${LOG_DIR}"

# Run main
if __name__ == "__main__":
    exec(open("train_models_radar_mix.py").read())
EOF
)

# Run the training with output logging
LOG_FILE="${LOG_DIR}/training_output_${TIMESTAMP}.log"
echo "Training log will be saved to: ${LOG_FILE}"
echo ""

# Run training and tee output to both terminal and log file
python -u "${SCRIPT_PATH}" 2>&1 | tee "${LOG_FILE}"

# Capture exit status
EXIT_STATUS=${PIPESTATUS[0]}

# =====================
# Post-Processing
# =====================
echo ""
echo "==========================================="
if [ ${EXIT_STATUS} -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: ${EXIT_STATUS}"
fi
echo "End Time: $(date)"
echo "Results saved to: ${LOG_DIR}"
echo "==========================================="

# Generate summary report
SUMMARY_FILE="${LOG_DIR}/experiment_summary.txt"
cat > "${SUMMARY_FILE}" <<EOF
Radar Training Experiment Summary
==================================
Timestamp: ${TIMESTAMP}
Exit Status: ${EXIT_STATUS}

Hardware:
  GPU Used: ${GPU_ID}
  
Dataset:
  Data Root: ${DATA_ROOT}

Hyperparameters:
  Window Size: ${WINDOW_SIZE}
  Max Points: ${MAX_POINTS}
  Batch Size: ${BATCH_SIZE}
  Epochs: ${NUM_EPOCHS}
  Learning Rate: ${LEARNING_RATE}
  Num Workers: ${NUM_WORKERS}

Output:
  Log Directory: ${LOG_DIR}
  Training Log: ${LOG_FILE}
EOF

echo "Summary report created at ${SUMMARY_FILE}"

# Extract final metrics from log if training succeeded
if [ ${EXIT_STATUS} -eq 0 ]; then
    echo ""
    echo "=== Final Training Results ==="
    grep -E "Epoch ${NUM_EPOCHS}|Accuracy|F1-Score|Recall|Precision|AUC-ROC" "${LOG_FILE}" | tail -6
fi

exit ${EXIT_STATUS}
