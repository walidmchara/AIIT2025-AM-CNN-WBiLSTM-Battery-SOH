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

## Environment

Python 3.10+ with PyTorch, NumPy and Pandas.

```bash
pip install -r requirements.txt
pytest -q
```

## Experiments

```bash
# Independent test on NASA battery B0018 (24 °C group)
python train.py --group 24C --epochs 20 --out outputs/B0018

# Independent test on NASA battery B0030 (43 °C group)
python train.py --group 43C --epochs 20 --out outputs/B0030

# Fast integration test: same real data, reduced neural-network width
python train.py --group 24C --smoke --out outputs/check

# CNN-BiLSTM control without wavelet activation or attention
python train.py --group 24C --model CNN-BiLSTM --epochs 20 --out outputs/CNN_BiLSTM
```

The training command saves a model checkpoint, training history, held-out-cell predictions and evaluation metrics. Validation, normalization and tuning never use the independent test cell.

## Evaluation scope

The model and data interface have automated tests. A new result is not equivalent to the publication's reported accuracy until the original feature extraction, optimizer settings, epochs and experimental rules are reconciled. [Architecture and validation notes](docs/ARCHITECTURE_AND_VALIDATION.md) document these distinctions. The closely related *Clean Energy* journal version has its own repository and experimental configuration.
