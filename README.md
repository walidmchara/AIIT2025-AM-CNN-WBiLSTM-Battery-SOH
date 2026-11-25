# AM–CNN–WBiLSTM for Lithium-Ion Battery SOH Estimation

**Publication:** W. Mchara et al., *Smart Health State Diagnosis of Lithium-Ion Batteries Using Wavelet-Enhanced Hybrid Deep Learning with an Attention Mechanism*. International Conference on Innovation in Artificial Intelligence and Internet of Things (AIIT), 2025. [DOI: 10.1109/AIIT63112.2025.11082918](https://doi.org/10.1109/AIIT63112.2025.11082918).

## Method

The hybrid architecture combines three convolutional feature-extraction blocks, a Morlet-enhanced bidirectional LSTM and temporal attention for State-of-Health regression. The published configuration uses convolutional filters 32–64–128 (kernel size 3 and pooling size 2), dropout 0.3, a 156-unit recurrent layer and 312 attention units. The code preserves these settings in [configs/paper.json](configs/paper.json), while documenting choices not completely specified in the article.

The real-valued Morlet function is `cos(2.1*x)·exp(-x²/3)`. It is applied to recurrent representations; the original conference paper does not define full modified LSTM gate equations. The implementation supports CNN-LSTM, CNN-BiLSTM, attention-based variants, and the proposed wavelet-enhanced model.

## Dataset and study design

The repository includes discharge-cycle measurements derived from the NASA PCoE lithium-ion battery-aging collection:

| Experiment | Development batteries | Independent test battery |
| --- | --- | --- |
| 24 °C group | B0005, B0006, B0007 | B0018 |
| 43 °C group | B0029, B0031, B0032 | B0030 |

The eight CSV datasets are under [data/processed/nasa/](data/processed/nasa/). Source metadata and checksums appear in [data/processed/manifest.json](data/processed/manifest.json). The original high-frequency MATLAB traces are not included; these tables contain discharge-cycle aggregates. The SOH label is derived from per-cell capacity relative to the first observed discharge, and predictor features exclude capacity and current-cycle SOH values.

## Quick start

Requires Python 3.10 or newer. The repository already includes the eight cycle-level NASA CSV files, so no account, credentials, data download, or external service is needed. From the repository directory:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python train.py --group 24C --smoke --out outputs/check
```

The smoke command runs two reduced-width epochs to check the data/model path. For the configured full-width experiment (20 epochs by default):

```bash
python train.py --group 24C --out outputs/B0018
```

To run the other independent held-out-cell experiment:

```bash
python train.py --group 43C --out outputs/B0030
```

Each run saves `checkpoint.pt`, `training_history.csv`, `test_predictions.csv`, and `metrics.json` under its output directory. To use another dataset, pass `--data-dir` pointing to a directory of CSVs named for the battery IDs in the selected experiment; columns and feature requirements are described in [data/README.md](data/README.md). Different battery IDs also require updating the experiment split in `configs/paper.json`.

## Experiments and paper correspondence

The default model is the AM-CNN-WBiLSTM with the article's listed CNN widths, kernel/pool sizes, dropout, recurrent width, attention width, and Morlet expression; both Morlet parameters are read from `configs/paper.json`. The article does not fully specify the modified LSTM gates or all experiment protocol details; the Morlet-on-standard-BiLSTM implementation and protocol choices are documented in [configs/paper.json](configs/paper.json) and [the validation notes](docs/ARCHITECTURE_AND_VALIDATION.md). As a result, this repository supports transparent, independent runs on the bundled cycle aggregates; it does not claim to exactly reproduce the paper's reported scores.

An ablation can be selected with `--model`, for example `--model CNN-BiLSTM`. Validation, normalization and tuning never use the independent test cell.

The paper itself is identified by its [DOI](https://doi.org/10.1109/AIIT63112.2025.11082918); no full-text copy or original experiment scripts are included here. 
