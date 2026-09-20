# Changelog

All notable changes to this repository are recorded here. Dataset access is handled separately on the [dataset portal](https://aiot-public-dataset-cuhk-x.cuhkaiot.com/).

## [Unreleased]

### Changed
- README and project website now present the CUHK-X Challenge as an official competition of [UbiComp/ISWC 2026](https://www.ubicomp.org/ubicomp-iswc-2026/cuhk-x-competition/): Kaggle phase closed 2026-09-15, verification 2026-09-19 to 09-30, Grand Finals 2026-10-12 in Shanghai. The README gains a challenge section with both tracks, the timeline, the finals schedule and the challenge-data links.

### Milestones
- 2026-09: CUHK-X Challenge listed as an official UbiComp/ISWC 2026 competition; Kaggle leaderboard frozen, verification under way.

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

[Unreleased]: https://github.com/openaiotlab/CUHK-X/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/openaiotlab/CUHK-X/releases/tag/v1.0.0
