# Changelog

All notable changes to this repository are recorded here. Dataset access is handled separately on the [dataset portal](https://aiot-public-dataset-cuhk-x.cuhkaiot.com/).

## [1.0.0] - 2026-09-07

First tagged release.

### Added
- Small-model HAR baselines (`SM/`) for RGB, IMU, mmWave radar and skeleton.
- CUHK-X-VLM (`LM/`): large-model benchmark code for the HAU and HARn tasks (action selection, captioning, emotion analysis, sequential reordering, next-action prediction).
- Project website (`docs/`): home, dataset, benchmarks and publications pages.
- CUHK-X Observatory (`docs/explorer/`): in-browser depth ⊕ IR ⊕ thermal playback, 40-action atlas and next-action reasoning quiz, with the asset build script in `scripts/`.
- Seven-modality alignment banner (`docs/images/cuhk-x-modalities.gif`) and the script that renders it.
- `CITATION.cff` so GitHub offers "Cite this repository".

### Milestones
- 2026-06: CUHK-X Challenge live on Kaggle (USD 20K prize pool, finals at UbiComp 2026).
- 2026-02: Paper accepted at MobiSys 2026.
- 2025-11: Best Presentation Award, ANAI Workshop @ MobiCom 2025.

[1.0.0]: https://github.com/openaiotlab/CUHK-X/releases/tag/v1.0.0
