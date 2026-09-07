# 📱 IMU Data Training

> Training pipeline for IMU-based action recognition using accelerometer, gyroscope, and magnetometer data.

---

## 🔧 Data Preparation

Run the following notebooks **in order**:

| Step | Notebook | Description |
|:----:|----------|-------------|
| 1️⃣ | `data_reader.ipynb` | Load and parse raw IMU data |
| 2️⃣ | `dataset_maker.ipynb` | Create train/test splits |
| 3️⃣ | `data_analysis.ipynb` | Visualize and analyze data |

---

## 🚀 Training

### Cross-Trial

```bash
bash ./command_accgyrmag_transformer_crosstrail.sh
```
📂 **Log**: `./activity_40/cnn_transformer/acc_gyr_mag/bsz128_all`

---

### Cross-User

```bash
bash ./command_activity20_accgyrmag_transformer_crossuser.sh
```
📂 **Log**: `./runs/activity_20/cross_user/cnn_transformer/acc_gyr_mag/bsz128`

---

### Cross-User (Resampled)

```bash
bash ./command_activity20_accgyrmag_transformer_crossuser.sh
```
📂 **Log**: `./runs/activity_20/cross_user/cnn_transformer/acc_gyr_mag/bsz128_resample_all`

---

## 📊 Dataset Info

| Dataset | Description |
|---------|-------------|
| `example data` | Sample data for testing |
| `data_imu` | Full IMU dataset |

> ℹ️ The index CSVs under `data_imu/` and `dataset/` cover the publicly released participants only (users 1–9 and 16–24). Held-out Challenge participants are intentionally excluded.
