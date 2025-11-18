# Data Change Log

<!-- This file tracks all dataset-related changes including new data, corrections, schema updates, and quality improvements -->
<!-- Follow semantic versioning for dataset: MAJOR.MINOR.PATCH -->

## Template
```
## [vX.Y.Z] - [YYYY-MM-DD]
### Type: [New Data/Data Correction/Schema Update/Quality Fix/Annotation Update]
### Summary
Brief description of data changes

### Data Statistics
- Total samples: [before → after]
- New samples added: [count]
- Samples removed: [count]
- Samples modified: [count]
- Data size: [GB] ([+/- change])

### Changes by Category
#### New Data
- Category/Task: [name]
  - Sample count: [N]
  - Source: [collection method/device]
  - Format: [file type, resolution, etc.]
  - Location: `data/path/to/new/data/`

#### Schema Updates
- File: `meta/session.json`
  - Added fields: [`field_name`] - purpose
  - Modified fields: [`field_name`] - what changed
  - Deprecated fields: [`field_name`] - migration path

#### Quality Improvements
- Fixed: [issue type, e.g., timestamp misalignment]
  - Affected samples: [count or list]
  - Method: [how it was fixed]
  - Validation: [how you verified the fix]

### Annotations
- New labels: [label names] - [count]
- Corrected labels: [count] samples
- Inter-annotator agreement: [metric if applicable]

### Data Split Updates
- Train/Val/Test: [XX%/YY%/ZZ%] or [counts]
- LOSO folds: [updated/unchanged]
- Rationale: [why splits changed]

### Backward Compatibility
- Breaking changes: [Yes/No] - [explain]
- Migration notes: [steps to update existing code]
- Deprecated paths: [old → new]

### Quality Assurance
- Validation checks: [list checks performed]
- Error rate: [P95/P99 metrics]
- Manual review: [N% sampled]
- Issues found: [link to issue tracker]

### Checksums & Downloads
- Dataset version: vX.Y.Z
- Download link: [URL]
- SHA-256: [hash]
- Mirror: [alternative URLs]
```

---

## Example Entry

## [v1.1.0] - 2025-11-18
### Type: New Data + Schema Update
### Summary
Added IMU sensor data for 12 action classes and fixed timestamp alignment issues in video frames.

### Data Statistics
- Total samples: 8,450 → 11,570
- New samples added: 3,120 IMU segments
- Samples removed: 0
- Samples modified: 145 (timestamp corrections)
- Data size: 156 GB (+18 GB)

### Changes by Category
#### New Data
- Modality: IMU (Accelerometer + Gyroscope)
  - Sample count: 3,120 segments
  - Source: Collected via smartphone IMU at 100Hz
  - Format: CSV with columns [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  - Location: `data/imu/raw/`

#### Schema Updates
- File: `meta/session.json`
  - Added fields: [`subject_age_group`] - Optional field for demographic analysis (values: "18-25", "26-35", "36-50", "50+")
  - Added fields: [`imu_sampling_rate`] - Records IMU sampling frequency in Hz

#### Quality Improvements
- Fixed: Frame timestamp misalignment in video data
  - Affected samples: 145 video sequences
  - Method: Re-extracted timestamps from video metadata using FFmpeg
  - Validation: Cross-modal alignment error P95 < 20ms verified on 1% sample

### Annotations
- No new labels added
- Corrected labels: 23 samples (action boundary refinement)

### Data Split Updates
- LOSO folds: Updated to balance action class distribution
- Train/Val/Test: Maintained 70%/15%/15%
- Rationale: New IMU data required rebalancing to prevent class imbalance in cross-validation

### Backward Compatibility
- Breaking changes: No
- Migration notes: Existing RGB-only pipelines work without changes. To use IMU data, update data loader to read from `data/imu/`
- Deprecated paths: None

### Quality Assurance
- Validation checks: Timestamp continuity, sensor range validation, sync verification
- Error rate: Cross-modal alignment P95 < 20ms, P99 < 35ms
- Manual review: 1% sampled, all passed
- Issues found: None

### Checksums & Downloads
- Dataset version: v1.1.0
- Download link: https://datasets.example.com/cuhk-x/v1.1.0
- SHA-256: Available in `releases/v1.1.0/checksums.txt`
- Mirror: Asia node available with resumable downloads

---

## [vX.Y.Z] - [YYYY-MM-DD]
### Type: 
### Summary


### Data Statistics
- Total samples: 
- New samples added: 
- Samples removed: 
- Samples modified: 
- Data size: 

### Changes by Category


### Backward Compatibility


### Quality Assurance


### Checksums & Downloads

