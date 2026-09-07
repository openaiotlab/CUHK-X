#!/bin/bash
#depth2/6/7/11/20/21/32/34/36/37
#rgb679 11 12 20 21  32  36 37 
# simclr_train_multiple.sh

# setting
DATASET_PATH="/YOUR/PATH/TO/CUHKX/SM/Data/RGB"
SCRIPT_PATH="simclr_10.py"  
LOG_DIR="training_logs_contra_lt"
NUM_USERS=30
BATCH_SIZE=64
EPOCHS=20
LR=1e-4
TEMPERATURE=0.5
IMAGE_SIZE=112

# build log directory
mkdir -p "$LOG_DIR"

# training users list
USERS=("user1" "user7" "user13" "user20" "user26")
SUP_ACTIONS="6 7 9 11 12 20 21  32  36 37"
# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo " Starting training multiple users - Timestamp: $TIMESTAMP"
echo " User list: ${USERS[@]}"
echo " Log directory: $LOG_DIR"
echo "========================================"

# Iterate over each user for training
for USER in "${USERS[@]}"; do
    echo " starting training: $USER"
    
    # Create a user-specific log filename (including user ID and timestamp)
    LOG_FILE="$LOG_DIR/${USER}_$(date +"%Y%m%d_%H%M%S").log"
    
    echo "========================================"
    echo "Starting training for $USER with supervised actions: $SUP_ACTIONS"
    echo "Log: $LOG_FILE"
    echo "========================================"

    python simclr_10.py \
        --test_user "$USER" \
        --dataset_path "$DATASET_PATH" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --temperature "$TEMPERATURE" \
        --image_size "$IMAGE_SIZE" \
        --supervised_action_ids $SUP_ACTIONS \
        > "$LOG_FILE" 2>&1

    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo " User $USER training completed! Log saved to: $LOG_FILE"
    else
        echo " User $USER training failed! Check log: $LOG_FILE"
    fi
    
    echo "========================================"
done

echo " All users training completed!"
echo " Total users trained: ${#USERS[@]}"
echo " Log files located at: $LOG_DIR/"
echo " Completion time: $(date)"