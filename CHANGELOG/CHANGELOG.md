# CUHK-X Changelog
<!-- 本文件遵循 Keep a Changelog 与语义化版本（SemVer）理念；数据/评测/代码分层版本分别打 Tag。 -->
<!-- 发布日期为 UTC，若无特殊说明，时间界定均为 AoE 截止。 -->


## [v1.1.0] - 2025-11-118
- dataset: v1.1.0
- evalspec: v1.0
- code: v0.4.0

### Highlights
- 新增 IMU 子集并完善时间对齐；发布多模态基线，LOSO Top‑1 提升 +1.8pp。
- 官网新增「新人 30 分钟上手」，下载失败率从 2.1% 降至 0.6%。

### Added / Changed / Fixed
- [data] 新增 3,120 段 IMU 片段（覆盖 12 类动作）；补齐丢帧的时间戳。
- [schema] `meta/session.json` 新增 `subject_age_group`（可选），用于统计分析。
- [split] 更新 LOSO v1.1 划分，平衡各 fold 的动作类别分布。
- [benchmark] HAU 增加 SPICE 指标；评测脚本兼容旧报告格式。
- [baseline] 发布 MM‑Tiny（RGB+IMU）；LOSO Top‑1=62.3%，Top‑5=88.4%（seed=42）。
- [code] `eval.py` 新增 `--modalities rgb,imu`；提升评测吞吐 +15%。
- [docs] 新增快速上手与 FAQ；补充常见错误码对照。
- [infra] 上线亚洲镜像节点；提供断点续传与 SHA‑256 校验。
- [privacy] 对测试集人脸区域做弱化处理（模糊半径 9px）；不影响关键指标。
- [deprecate] 标记 `eval_legacy.py` 为弃用，计划在 v1.3 移除。

### Backward Compatibility & Migration
- 无破坏性变更。若使用自研解析器，请忽略或兼容新增字段。
- 建议升级步骤：更新划分文件→拉取 code/v0.4.0→按文档重跑基线核对结果。

### Baseline & Metrics Changes
- RGB 单模态基线 Top‑1：55.0%（+0.9pp）
- RGB+IMU 多模态基线 Top‑1：62.3%（新）
- 复现实验：`python eval.py --task har --modalities rgb,imu --seed 42`

### QA & Verification
- 1% 抽样人工复核，跨模态对齐误差 P95 < 20ms。
- 全量评测脚本 8/8 通过；日志与报告归档于 `artifacts/2025-12-15/`。

### Known Issues
- 在 macOS M2 上安装雷达依赖需额外步 骤；见 FAQ#5。

### Download & Checksums
- checksums: `releases/v1.1.0/checksums.txt`