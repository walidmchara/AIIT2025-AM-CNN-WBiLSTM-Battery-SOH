"""Battery-independent sequence windows and training-only normalization."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch


@dataclass
class BatteryDataset:
    train: tuple[torch.Tensor, torch.Tensor]
    val: tuple[torch.Tensor, torch.Tensor]
    test: tuple[torch.Tensor, torch.Tensor]
    test_cycles: np.ndarray
    test_cell: str
    train_cells: list[str]
    metadata: dict


def read_battery(path: Path, features: list[str]):
    df = pd.read_csv(path).sort_values('cycle', kind='stable').reset_index(drop=True)
    required = features + ['capacity_ah', 'cycle']
    missing = [x for x in required if x not in df]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    if len(df) < 16:
        raise ValueError(f'{path}: too few discharge cycles')
    if df['cycle'].duplicated().any():
        raise ValueError(f'{path}: duplicated cycle indices')
    numeric = df[features + ['capacity_ah']].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric['capacity_ah'].to_numpy()).all() or (numeric['capacity_ah'] <= 0).any():
        raise ValueError(f'{path}: invalid cycle-level capacities')
    initial = float(numeric['capacity_ah'].iloc[0])
    y = numeric['capacity_ah'].to_numpy(dtype='float32') / initial
    x = numeric[features].to_numpy(dtype='float32')
    return x, y, df['cycle'].to_numpy(), initial


def build_experiment(data_dir: Path, train_cells: list[str], test_cell: str,
                     features: list[str], lookback: int = 16,
                     validation_fraction: float = .20) -> BatteryDataset:
    if len(train_cells) == 0 or test_cell in train_cells or len(set(train_cells)) != len(train_cells):
        raise ValueError('Independent train and test battery IDs are required')
    if not (0.05 <= validation_fraction < .5):
        raise ValueError('Validation fraction must be in [0.05, 0.5)')
    if lookback < 8:
        raise ValueError('The three CNN pooling layers require lookback >= 8')
    data = {c: read_battery(data_dir / f'{c}.csv', features) for c in train_cells + [test_cell]}
    cutoffs = {c: int(len(data[c][0]) * (1 - validation_fraction)) for c in train_cells}
    for c, stop in cutoffs.items():
        if stop <= lookback or len(data[c][0]) - stop < 1:
            raise ValueError(f'Insufficient cycles for train/validation in {c}')
    # Fit medians and standardization exclusively on development-cell TRAIN segments.
    observed = np.concatenate([data[c][0][:cutoffs[c]] for c in train_cells], axis=0)
    medians = np.nanmedian(np.where(np.isfinite(observed), observed, np.nan), axis=0)
    if not np.isfinite(medians).all():
        raise ValueError('A selected feature is entirely missing in the training cells')
    finite_train = np.where(np.isfinite(observed), observed, medians)
    mu = finite_train.mean(0)
    scale = finite_train.std(0)
    scale = np.where(scale > 1e-7, scale, 1.0)

    sample_sets = {'train': [], 'val': [], 'test': []}
    labels = {'train': [], 'val': [], 'test': []}
    test_cycles = []
    for c, (x, y, cycles, _) in data.items():
        x = np.where(np.isfinite(x), x, medians)
        x = (x - mu) / scale
        for target_idx in range(lookback, len(x)):
            role = ('test' if c == test_cell
                    else 'train' if target_idx < cutoffs[c] else 'val')
            # Predictor window ends at target_idx-1; the target cycle is held out.
            sample_sets[role].append(x[target_idx - lookback:target_idx])
            labels[role].append(y[target_idx])
            if role == 'test':
                test_cycles.append(cycles[target_idx])
    tensors = {}
    for role in ('train', 'val', 'test'):
        if not sample_sets[role]:
            raise ValueError(f'Empty {role} partition')
        tensors[role] = (torch.tensor(np.stack(sample_sets[role]), dtype=torch.float32),
                         torch.tensor(labels[role], dtype=torch.float32))
    return BatteryDataset(tensors['train'], tensors['val'], tensors['test'],
                          np.asarray(test_cycles), test_cell, train_cells, {
                              'train_cells': train_cells, 'test_cell': test_cell,
                              'lookback': lookback, 'features': features,
                              'validation_last_fraction': validation_fraction,
                              'training_cell_cutoffs': cutoffs,
                              'train_feature_medians': medians.tolist(),
                              'train_feature_means': mu.tolist(),
                              'train_feature_stds': scale.tolist(),
                              'soh_target': 'capacity_ah / first_observed_capacity_ah of same cell',
                              'target_cycle_is_never_in_window': True,
                              'experimental_scope': 'cycle aggregates rather than full NASA MATLAB signals'
                          })
