# MedFM-Bench

**An experimental framework for reproducible 3D medical image segmentation.**

MedFM-Bench is a research codebase being developed toward systematic evaluation of medical imaging models. The current implementation establishes a 3D brain-tumour segmentation pipeline using the **Medical Segmentation Decathlon (MSD), Task01_BrainTumour** dataset.

> **Current scope:** the committed implementation contains a MONAI 3D U-Net baseline, data handling and validation utilities, patient-level data partitions, patch-based training, and full-volume validation. It does **not yet** implement a comparative benchmark of multiple medical foundation models. Foundation-model integration and cross-model evaluation remain future work.

## Project overview

The current pipeline makes the data and evaluation stages explicit:

1. Discover labelled 3D MRI volumes and their segmentation masks.
2. Load a fixed patient-level train/validation/internal-test partition.
3. Check partition integrity and prevent patient overlap.
4. Normalize each MRI channel independently using foreground-based Z-score normalization.
5. Extract 3D training patches with foreground-aware sampling.
6. Train a 3D U-Net segmentation baseline.
7. Evaluate validation patients using full-volume sliding-window inference.
8. Record training history, validation metrics, and model checkpoints.

The repository is structured so that additional model implementations and evaluation protocols can be added as the project develops.

## Dataset

The current experiment uses **MSD Task01_BrainTumour**.

- Official dataset page: [Medical Segmentation Decathlon — Task01_BrainTumour](https://medicaldecathlon.com/dataaws/)
- Input: four-channel 3D MRI volumes in NIfTI format.
- Target: multi-class brain-tumour segmentation.
- Labelled training cases available in the task: 484.
- The repository uses the labelled `imagesTr/` and `labelsTr/` cases. It does not currently use the official unlabelled `imagesTs/` cases.

The repository includes a fixed patient-level partition of the 484 labelled cases:

| Partition | Patients | Use in the current pipeline |
|---|---:|---|
| Training | 387 | Model fitting |
| Validation | 48 | Model selection and validation metrics |
| Internal test | 49 | Reserved; not used by the current training loop |
| **Total** | **484** | |

The partition is stored in `configs/splits/brats_seed42.json`. The training script checks for duplicate IDs, overlap between partitions, and consistency between the split file and the discovered dataset.

### Expected dataset layout

Download and prepare the dataset separately. The raw medical images are not stored in this GitHub repository.

```text
MedFM-Bench/
├── configs/
│   ├── config.yaml
│   └── splits/
│       └── brats_seed42.json
├── datasets/
│   └── raw/
│       └── Task01_BrainTumour/
│           ├── dataset.json
│           ├── imagesTr/
│           │   ├── BRATS_001.nii.gz
│           │   └── ...
│           └── labelsTr/
│               ├── BRATS_001.nii.gz
│               └── ...
├── scripts/
├── src/
├── tests/
└── requirements.txt
```

The dataset root must match `dataset.root_dir` in `configs/config.yaml`. The loader expects matching image and label filenames under `imagesTr/` and `labelsTr/`.

Please follow the dataset provider's terms and cite the Medical Segmentation Decathlon when using the data. Do not upload the dataset or patient volumes to this repository.

## Current model and training setup

The current baseline is a **3D U-Net implemented with MONAI**.

| Component | Current configuration |
|---|---|
| Architecture | MONAI 3D U-Net |
| Input channels | 4 |
| Output channels | 4 |
| Channel widths | 16, 32, 64, 128 |
| Residual units | 2 |
| Normalization | Instance normalization |
| Training patch size | 96 × 96 × 96 |
| Batch size | 1 |
| Maximum epochs | 50 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Loss | Dice + Cross-Entropy (MONAI `DiceCELoss`) |
| Mixed precision | Enabled in the default configuration |
| Random seed | 42 |

### Preprocessing and patch sampling

- MRI channels are normalized independently using Z-score statistics computed over non-zero foreground voxels.
- Image tensors are converted from `(H, W, D, C)` to `(C, H, W, D)`.
- Segmentation labels are retained as integer class IDs.
- Training patches are extracted at `96 × 96 × 96`.
- With the default `foreground_prob: 0.70`, patch selection is foreground-centred when a foreground class is available; otherwise, a spatially uniform crop is sampled.
- Foreground-centred sampling considers label IDs 1, 2, and 3.
- Random spatial flips are applied consistently to the image and label.

### Validation protocol

Validation uses full 3D patient volumes and MONAI sliding-window inference, with a default ROI of `96 × 96 × 96`, Gaussian blending, and an overlap of `0.25`.

The validation code reports Dice scores for these regions:

| Region | Included label IDs |
|---|---|
| Whole Tumour (WT) | 1, 2, 3 |
| Tumour Core (TC) | 2, 3 |
| Enhancing Tumour (ET) | 3 |

The validation score used for model selection is the mean of the WT, TC, and ET Dice scores. ET Dice is averaged over ET-positive validation patients; ET-negative patients are tracked separately using false-positive ET voxel counts.

The current training loop evaluates the **validation partition**. The 49-patient internal test partition is reserved and is not used for training or model selection.

## Installation

Use a Python environment with a PyTorch build compatible with your CUDA setup for full training.

```bash
git clone https://github.com/ZahraAlipour703/MedFM-Bench.git
cd MedFM-Bench
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

Install the pinned project requirements and the configuration dependency used by the training runner:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install omegaconf
```

If you use a custom PyTorch/CUDA installation, make sure the installed PyTorch build is compatible with your GPU driver and CUDA environment.

## Running the pipeline

Run commands from the repository root.

### 1. Run preflight checks

Preflight checks validate the dataset partitions, load and inspect a training patch, and construct the model. The preflight can run on CPU or CUDA:

```bash
python -m scripts.train --preflight
```

You can explicitly provide the configuration file:

```bash
python -m scripts.train --preflight --config configs/config.yaml
```

### 2. Start training

Full training currently requires a CUDA-enabled GPU. The runner intentionally stops if CUDA is unavailable.

```bash
python -m scripts.train
```

To use a different configuration:

```bash
python -m scripts.train --config path/to/config.yaml
```

The default configuration writes experiment artifacts beneath:

```text
results/brats_unet3d_seed42/
└── run_<timestamp>/
    ├── config.yaml
    ├── history.csv
    ├── best_model.pt
    └── latest_checkpoint.pt
```

The run history records training loss, validation score, WT/TC/ET Dice, ET-negative false-positive counts, learning rate, and epoch duration. Raw data, checkpoints, and experiment outputs are excluded from version control by `.gitignore`.

## Validation and audit scripts

The repository includes scripts for checking the data pipeline and individual stages:

| Script | Purpose |
|---|---|
| `scripts/validate_dataset.py` | Validate the MSD dataset structure and image/label consistency |
| `scripts/test_dataset.py` | Inspect dataset loading and sample tensor properties |
| `scripts/test_split.py` | Generate and inspect patient-level partitions |
| `scripts/test_patch_dataset.py` | Smoke-test 3D patch extraction and label consistency |
| `scripts/audit_patch_sampling.py` | Audit patch sampling across the training partition |
| `scripts/test_brats_training_step.py` | Exercise one real optimization step using a dataset patch |
| `scripts/test_brats_full_volume_inference.py` | Exercise full-volume sliding-window inference on a validation case |

For example, run a patch-extraction smoke test with:

```bash
python -m scripts.test_patch_dataset
```

These scripts provide stage-level checks; their presence should not be interpreted as evidence that a complete training run or final benchmark evaluation has been completed.

## Repository structure

```text
MedFM-Bench/
├── configs/
│   ├── config.yaml                 # Experiment configuration
│   └── splits/
│       └── brats_seed42.json       # Fixed patient-level partition
├── scripts/
│   ├── train.py                    # Preflight and training entry point
│   ├── audit_patch_sampling.py
│   ├── validate_dataset.py
│   └── test_*.py                   # Data, training, and inference checks
├── src/
│   ├── data/                       # 3D patch dataset
│   ├── datasets/                   # BraTS loader, subsets, split and validation
│   ├── engine/                     # Training and full-volume validation
│   ├── models/                     # 3D U-Net baseline
│   ├── registry/                   # Dataset/model registries
│   ├── transforms/                 # MRI normalization and tensor conversion
│   └── utils/                      # Seeding, device and logging utilities
├── tests/                          # Unit-level tests
├── requirements.txt
└── README.md
```

## Current status and limitations

This repository is an **ongoing research implementation**, not a completed clinical system or a finished multi-model benchmark.

- The current committed model implementation is a 3D U-Net baseline.
- Multiple medical foundation-model backbones and systematic cross-model comparisons are not yet implemented.
- The current training script uses the training and validation partitions; the internal test partition remains reserved.
- No final held-out test results or completed benchmark comparison are reported in this repository snapshot.
- The code is intended for research and experimentation, not for clinical diagnosis or treatment decisions.

## Dataset citation

If you use MSD Task01_BrainTumour, please cite the dataset and follow its terms:

> Antonelli, M. et al. *The Medical Segmentation Decathlon*. Nature Communications, 13, 4128 (2022).  
> https://doi.org/10.1038/s41467-022-30695-9

Dataset access: [Medical Segmentation Decathlon — Data](https://medicaldecathlon.com/dataaws/).

## License

A license for this repository's code has not yet been added. Until a repository license is provided, do not assume that the source code is available for reuse, modification, or redistribution. The dataset's license is separate from the license of this codebase.

## Author

**Zahra Alipour**  
GitHub: [@ZahraAlipour703](https://github.com/ZahraAlipour703)
