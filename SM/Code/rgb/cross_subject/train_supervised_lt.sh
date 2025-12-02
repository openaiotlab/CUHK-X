#!/bin/bash

# train_supervised_users.sh - 

# parameters for supervised training on specific users and actions
SCRIPT_PATH="train_supervised_lt.py"  
DATASET_PATH="/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB"
LOG_DIR="supervised_logs_lt"
NUM_USERS=30
BATCH_SIZE=64
EPOCHS=20
LR=1e-4
IMAGE_SIZE=224

# test users and supervised action IDs
USERS=("user1" "user7" "user13" "user20" "user26")
SUP_ACTIONS="6 7 9 11 12 20 21 32 36 37"

# create log directory
mkdir -p "$LOG_DIR"

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo " Starting supervised training - Timestamp: $TIMESTAMP"
echo " Users: ${USERS[@]}"
echo " Supervised Action IDs: $SUP_ACTIONS"
echo " Log Directory: $LOG_DIR"
echo "========================================"

# Iterate over each user for training
for USER in "${USERS[@]}"; do
    echo " Starting training for user: $USER"
    
    # Create user-specific log file name (including user ID and timestamp)
    LOG_FILE="$LOG_DIR/${USER}_$(date +"%Y%m%d_%H%M%S").log"
    
    echo " Log File: $LOG_FILE"
    echo " Training Start Time: $(date)"
    echo " Training Command: python $SCRIPT_PATH --dataset_path $DATASET_PATH --test_user $USER --num_users $NUM_USERS --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --image_size $IMAGE_SIZE --supervised_action_ids $SUP_ACTIONS"
    
    # Run training command, redirecting both stdout and stderr to the log file
    if command -v unbuffer &> /dev/null; then
        unbuffer python "$SCRIPT_PATH" \
            --dataset_path "$DATASET_PATH" \
            --test_user "$USER" \
            --num_users "$NUM_USERS" \
            --batch_size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --image_size "$IMAGE_SIZE" \
            --supervised_action_ids $SUP_ACTIONS \
            > "$LOG_FILE" 2>&1
    else
        python "$SCRIPT_PATH" \
            --dataset_path "$DATASET_PATH" \
            --test_user "$USER" \
            --num_users "$NUM_USERS" \
            --batch_size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --image_size "$IMAGE_SIZE" \
            --supervised_action_ids $SUP_ACTIONS \
            > "$LOG_FILE" 2>&1
    fi
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo " User $USER training completed! Log saved to: $LOG_FILE"
        
        # Extract final accuracy from log (optional)
        FINAL_ACC=$(grep -o "Test Acc: [0-9.]*%" "$LOG_FILE" | tail -1)
        if [ -n "$FINAL_ACC" ]; then
            echo " Final Accuracy: $FINAL_ACC"
        fi
    else
        echo " User $USER training failed! Please check the log: $LOG_FILE"
    fi
    
    echo "========================================"
done

echo " All users training completed!"
echo " Total users trained: ${#USERS[@]}"
echo " Log files located at: $LOG_DIR/"
echo " Completion time: $(date)"

# Generate training summary
SUMMARY_FILE="$LOG_DIR/supervised_summary_$TIMESTAMP.txt"
echo "Supervised Training Summary - $TIMESTAMP" > "$SUMMARY_FILE"
echo "================================" >> "$SUMMARY_FILE"
for USER in "${USERS[@]}"; do
    USER_LOG=$(ls -t "$LOG_DIR"/${USER}_*.log 2>/dev/null | head -1)
    if [ -n "$USER_LOG" ]; then
        FINAL_ACC=$(grep -o "Test Acc: [0-9.]*%" "$USER_LOG" | tail -1)
        echo "$USER: $FINAL_ACC" >> "$SUMMARY_FILE"
    else
        echo "$USER: No log found" >> "$SUMMARY_FILE"
    fi
done
echo " Summary File: $SUMMARY_FILE"