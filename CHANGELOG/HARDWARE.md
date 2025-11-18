# Hardware & Infrastructure Change Log

<!-- This file tracks hardware setup, infrastructure changes, deployment updates, and system requirements -->
<!-- Include details about servers, devices, dependencies, and runtime environments -->

## Template
```
## [YYYY-MM-DD] - [Author/Team]
### Type: [Hardware Update/Infrastructure/Deployment/Environment/Dependencies]
### Summary
Brief description of hardware or infrastructure changes

### Hardware Changes
#### Devices
- Device: [name/model]
  - Purpose: [training/inference/data collection]
  - Specs: [CPU/GPU/RAM/Storage]
  - Quantity: [N units]
  - Location: [lab/server room/cloud]
  - Status: [added/upgraded/retired]

#### Sensors/Cameras
- Sensor: [type/model]
  - Modality: [RGB/depth/IMU/radar/etc.]
  - Resolution/Frequency: [specs]
  - Calibration: [date/method]
  - Issues: [known problems]

### Infrastructure Updates
#### Servers/Compute
- Server: [hostname/identifier]
  - CPU: [model, cores]
  - GPU: [model, count, VRAM]
  - RAM: [amount]
  - Storage: [type, capacity]
  - OS: [Linux distribution/version]
  - CUDA: [version]
  - Purpose: [workload type]

#### Network/Storage
- Change: [description]
  - Type: [network upgrade/storage expansion/backup]
  - Capacity: [bandwidth/storage size]
  - Impact: [performance improvement]

### Software Environment
#### System Requirements
- OS: [supported versions]
- Python: [version requirement]
- CUDA: [minimum version]
- cuDNN: [version]
- Driver: [minimum version]

#### Dependencies
- Added: [package==version] - reason
- Updated: [package: old_ver → new_ver] - reason
- Removed: [package] - reason
- Conflicts: [known compatibility issues]

### Deployment Changes
- Environment: [dev/staging/production]
- Changes: [what was deployed]
- Rollback plan: [procedure]
- Monitoring: [metrics to watch]

### Performance Impact
- Training speed: [before → after]
- Inference latency: [before → after]
- Throughput: [before → after]
- Resource utilization: [CPU/GPU/memory]

### Known Issues & Limitations
- Issue: [description]
  - Platform: [where it occurs]
  - Workaround: [temporary solution]
  - Status: [open/resolved/wontfix]

### Compatibility Notes
- Minimum requirements: [hardware/software specs]
- Tested configurations: [list verified setups]
- Unsupported: [known incompatibilities]

### Documentation & Setup
- Setup guide: [link to detailed instructions]
- Troubleshooting: [common issues and solutions]
- Contact: [who to reach for hardware issues]
```

---

## Example Entry

## [2025-11-18] - Infrastructure Team
### Type: Infrastructure + Dependencies
### Summary
Upgraded training server GPUs from RTX 3090 to A100, and updated CUDA/PyTorch stack. Added macOS M2 compatibility notes.

### Hardware Changes
#### Devices
- Device: NVIDIA A100 40GB
  - Purpose: Model training and evaluation
  - Specs: 40GB HBM2, 6912 CUDA cores, 432 Tensor cores
  - Quantity: 4 units
  - Location: Main server room (Node-03)
  - Status: Replaced RTX 3090 (24GB)

### Infrastructure Updates
#### Servers/Compute
- Server: train-01.lab.example.com
  - CPU: AMD EPYC 7763 (64 cores)
  - GPU: 4x NVIDIA A100 40GB (upgraded from 4x RTX 3090)
  - RAM: 512 GB DDR4
  - Storage: 8TB NVMe SSD RAID-0
  - OS: Ubuntu 22.04 LTS
  - CUDA: 12.1 (upgraded from 11.8)
  - Purpose: Multi-GPU distributed training

#### Network/Storage
- Change: Deployed Asia mirror node for dataset distribution
  - Type: CDN with resumable download support
  - Capacity: 100 Gbps bandwidth, 2TB cache
  - Impact: Download failure rate reduced from 2.1% to 0.6%

### Software Environment
#### System Requirements
- OS: Ubuntu 20.04+ / macOS 12+ / Windows 10+
- Python: 3.8 - 3.11 (3.12 not yet tested)
- CUDA: 11.8 or 12.1+ (for GPU acceleration)
- cuDNN: 8.6+
- Driver: NVIDIA 525+ (Linux), 527+ (Windows)

#### Dependencies
- Updated: `torch: 2.0.1 → 2.1.2` - Better A100 utilization, Flash Attention support
- Updated: `torchvision: 0.15.2 → 0.16.2` - Compatibility with torch 2.1.2
- Added: `accelerate==0.25.0` - Simplified multi-GPU training setup
- Note: macOS M2 users need special radar dependencies (see FAQ#5)

### Deployment Changes
- Environment: production training cluster
- Changes: Migrated all training jobs to new A100 nodes
- Rollback plan: RTX 3090 nodes kept online for 2 weeks as backup
- Monitoring: GPU utilization, training throughput, error rates

### Performance Impact
- Training speed: 100 epochs in 8h → 3.2h (2.5x faster)
- Inference latency: 45ms → 28ms per batch (batch_size=32)
- Throughput: 850 samples/sec → 2100 samples/sec
- Resource utilization: GPU memory usage 85% (well within 40GB)

### Known Issues & Limitations
- Issue: Radar processing dependencies fail on macOS M2
  - Platform: macOS with Apple Silicon (M1/M2/M3)
  - Workaround: Install using `arch -arm64 pip install` or use conda-forge channel
  - Status: Open, documented in FAQ#5
  - Reference: https://github.com/org/repo/issues/78

- Issue: CUDA 12.1 incompatible with old PyTorch < 2.0
  - Platform: All Linux/Windows systems
  - Workaround: Upgrade to PyTorch 2.1+
  - Status: Resolved in updated requirements.txt

### Compatibility Notes
- Minimum requirements: 
  - GPU: 8GB VRAM for inference, 16GB for training
  - CPU: 8 cores recommended for data loading
  - RAM: 16GB minimum, 32GB recommended
- Tested configurations:
  - ✅ Ubuntu 22.04 + A100 + CUDA 12.1 + PyTorch 2.1.2
  - ✅ Ubuntu 20.04 + RTX 3090 + CUDA 11.8 + PyTorch 2.0.1
  - ✅ macOS 13 (M2) + CPU-only + PyTorch 2.1.2
  - ✅ Windows 11 + RTX 4090 + CUDA 12.1 + PyTorch 2.1.2
- Unsupported: 
  - ❌ PyTorch 1.x (too old)
  - ❌ CUDA < 11.8
  - ❌ Python 3.7 (EOL)

### Documentation & Setup
- Setup guide: docs/setup/hardware_requirements.md
- Troubleshooting: docs/faq.md#hardware-issues
- Contact: infra-team@example.com for hardware provisioning

---

## [YYYY-MM-DD] - [Your Entry Here]
### Type: 
### Summary


### Hardware Changes


### Software Environment


### Performance Impact


### Known Issues & Limitations


### Compatibility Notes

