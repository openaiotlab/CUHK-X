#!/bin/bash

# ================== setting ==================
SCRIPT="train_supervised_44.py"  # ← Please ensure this filename matches your Python script
DATASET_PATH="/YOUR/PATH/TO/CUHKX/SM/Data/RGB"
LOG_DIR="logs_supervised_all_subject"

# Hyperparameters (consistent with code defaults, adjust as needed)
BATCH_SIZE=64
EPOCHS=20
LR=1e-4
IMAGE_SIZE=224

USERS=("user1" "user7" "user13" "user20" "user26")


mkdir -p "$LOG_DIR"

echo " Starting global 44-class supervised training"
echo " Dataset: $DATASET_PATH"
echo " Users: ${USERS[*]}"
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
        --image_size "$IMAGE_SIZE" \
        > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "  $USER training completed successfully."
    else
        echo "  $USER training failed! Check log: $LOG_FILE"
    fi

    echo "----------------------------------------"
done

echo "🎉 All users training finished!"
echo "📂 Log directory: $LOG_DIR"