from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from aiit_soh.data import build_experiment

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['voltage_mean','voltage_std','current_mean','current_std','temperature_mean','discharge_duration_s','ambient_temperature']


def test_real_nasa_splits_24c():
    d=build_experiment(ROOT/'data/processed/nasa',['B0005','B0006','B0007'],'B0018',FEATURES)
    assert len(d.test[1])==132-16
    assert d.test_cell not in d.train_cells
    assert d.test[0].shape[-1]==len(FEATURES)
    assert np.array_equal(d.test_cycles,np.arange(17,133))


def test_real_nasa_splits_43c():
    d=build_experiment(ROOT/'data/processed/nasa',['B0029','B0031','B0032'],'B0030',FEATURES)
    assert len(d.test[1])==40-16
    assert d.val[0].shape[0]>0


def test_independent_battery_required():
    with pytest.raises(ValueError):
        build_experiment(ROOT/'data/processed/nasa',['B0005'],'B0005',FEATURES)


def test_source_capacity_not_present_in_inputs():
    d=build_experiment(ROOT/'data/processed/nasa',['B0005','B0006','B0007'],'B0018',FEATURES)
    assert not any(c in d.metadata['features'] for c in ['capacity_ah','soh_first_observed','target'])
    assert d.metadata['target_cycle_is_never_in_window']


def test_training_stats_independent_of_held_out_cell(tmp_path):
    for name, k in [('A',1.0),('B',1.0),('C',1000.0)]:
        base=np.arange(40,dtype=float)
        table=pd.DataFrame({'cycle':base+1,'capacity_ah':2-.01*base,'voltage_mean':base+k})
        table.to_csv(tmp_path/f'{name}.csv',index=False)
    d=build_experiment(tmp_path,['A','B'],'C',['voltage_mean'],lookback=8)
    assert d.metadata['train_feature_means'][0]<40
    assert d.test[0].abs().mean()>10  # Held-out distribution was not used for scaling.
