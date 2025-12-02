#!/bin/bash

# ================== setting ==================
# Replace with your actual Python script filename
SCRIPT="simclr_10_remove_env.py"

# Dataset path
DATASET_PATH="/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB"
export CUDA_VISIBLE_DEVICES=2
# Log directory
LOG_DIR="training_logs_contra_lt_remove_env"
mkdir -p "$LOG_DIR"

# Hyperparameters (consistent with code defaults)
BATCH_SIZE=64
EPOCHS=20
LR=1e-4
TEMPERATURE=0.5
IMAGE_SIZE=224

# Test users list (covering two environments)
USERS=("user1" "user7" "user13" "user20" "user26")

# Supervised training action prefixes (commonly used 10 actions in RGB dataset)
SUP_ACTION_IDS="6 7 9 11 12 20 21 32 36 37"

echo " Starting Environment-Aware SimCLR Training"
echo " Dataset: $DATASET_PATH"
echo " Test Users: ${USERS[*]}"
echo " Supervised Actions: $SUP_ACTION_IDS"
echo " Logs will be saved to: $LOG_DIR"
echo "========================================"

# Iterate over each user for training
for USER in "${USERS[@]}"; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/${USER}_${TIMESTAMP}.log"

    echo "  Training on $USER | Log: $LOG_FILE"

    python "$SCRIPT" \
        --test_user "$USER" \
        --dataset_path "$DATASET_PATH" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --temperature "$TEMPERATURE" \
        --image_size "$IMAGE_SIZE" \
        --supervised_action_ids $SUP_ACTION_IDS \
        --exact_match \
        > "$LOG_FILE" 2>&1

    # Check if successful
    if [ $? -eq 0 ]; then
        echo " SUCCESS: $USER training completed."
    else
        echo " FAILED: $USER training failed! Check log: $LOG_FILE"
    fi

    echo "----------------------------------------"
done

echo " All experiments finished!"
echo " Final logs directory: $LOG_DIR"