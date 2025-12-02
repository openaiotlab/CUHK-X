#!/bin/bash

# Set dataset path (optional, omit if default path in script is correct)
DATASET_PATH="/aiot-nvme-15T-x2-hk01/siyang/CUHK-X-Final/SM_data/RGB"

# userlist for training
USERS=("user1" "user7" "user13" "user20" "user26")

# log directory (auto-create)
LOG_DIR="contra_allusers_logs"
mkdir -p "$LOG_DIR"

# Iterate over each user for training
for user in "${USERS[@]}"; do
    # Generate log file name with timestamp (to the second)
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/${user}_train_${TIMESTAMP}.log"

    echo "[$(date)] Starting training for $user. Logging to $LOG_FILE"

    # Run training script, redirecting both stdout and stderr to the log file and displaying on terminal
    python simclr_44.py \
        --test_user "$user" \
        --dataset_path "$DATASET_PATH" \
        --batch_size 64 \
        --epochs 20 \
        --lr 1e-4 \
        --temperature 0.5 \
        --image_size 224 \
        --num_users 30 \
        2>&1 | tee "$LOG_FILE"

    echo "[$(date)] Finished training for $user. Log saved to $LOG_FILE"
    echo "--------------------------------------------------"
done

echo "All users trained. Logs are in $LOG_DIR/"