"""Train independent held-out-cell SOH experiments from the AIIT 2025 paper."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from aiit_soh.data import build_experiment
from aiit_soh.model import make_model
from aiit_soh.metrics import evaluate
from training_recovery import EpochCheckpoint


def main():
    ap = argparse.ArgumentParser(description='AIIT 2025 battery SOH held-out-cell experiment')
    ap.add_argument('--group', choices=['24C', '43C'], default='24C')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/nasa')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/paper.json')
    ap.add_argument('--model', choices=['AM-CNN-WBiLSTM','AM-CNN-BiLSTM', 'AM-CNN-LSTM',
                                        'CNN-BiLSTM', 'CNN-LSTM'], default='AM-CNN-WBiLSTM')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--checkpoint-dir', type=Path, help='Persistent recovery directory')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='Fast architecture/data pipeline check with small hidden sizes')
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/latest')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    settings = dict(cfg['model'])
    protocol = dict(cfg['protocol_choices_not_reported_exactly_in_paper'])
    experiment = cfg['experiments'][args.group]
    seed = protocol['seed']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    if args.smoke:
        # The full published widths remain in paper.json; no published-results claim.
        settings['conv_filters'] = [8, 16, 24]
        settings['lstm_hidden'] = 12
        settings['attention_units'] = 24
        protocol['epochs'] = 2
    epochs = args.epochs if args.epochs is not None else protocol['epochs']
    if epochs < 1:
        raise ValueError('epochs must be >= 1')
    dataset = build_experiment(args.data_dir, experiment['train'], experiment['test'],
                               cfg['features'], lookback=protocol['lookback_cycles'],
                               validation_fraction=protocol['validation_last_fraction_per_development_cell'])
    model = make_model(args.model, features=len(cfg['features']),
                       conv_filters=tuple(settings['conv_filters']),
                       kernels=tuple(settings['conv_kernels']),
                       pools=tuple(settings['pool_sizes']),
                       lstm_hidden=settings['lstm_hidden'],
                       attention_units=settings['attention_units'],
                       dropout=settings['dropout'],
                       morlet_frequency=settings['morlet_frequency'],
                       morlet_variance_divisor=settings['morlet_variance_divisor'])
    opt = torch.optim.Adam(model.parameters(), lr=protocol['learning_rate'])
    loader = DataLoader(TensorDataset(*dataset.train), batch_size=protocol['batch_size'], shuffle=True)
    data_files = []
    for cell in experiment['train'] + [experiment['test']]:
        path = Path(args.data_dir) / f'{cell}.csv'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        data_files.append({'path': path.name, 'sha256': digest})
    recovery = EpochCheckpoint(args.checkpoint_dir or args.out / "recovery", {
        "cfg": cfg, "group": args.group, "model": args.model, "epochs": epochs,
        "smoke": args.smoke, "data": data_files,
    })
    stateful={"model":model,"optimizer":opt}
    best, best_weights = float('inf'), None
    log=[];start_epoch=0
    if args.resume:
        prev=recovery.restore(stateful,device="cpu")
        start_epoch=prev["epoch"];best=prev["best"]
        best_weights=prev["best_weights"];log=prev["log"]
        print(f"Resuming from epoch {start_epoch}")
    for epoch in range(start_epoch,epochs):
        model.train(); total=0; n=0
        for x,y in loader:
            opt.zero_grad(); pred=model(x)
            loss=(pred-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite model loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); total+=float(loss.item())*len(x); n+=len(x)
        model.eval()
        with torch.no_grad():
            val=float((model(dataset.val[0])-dataset.val[1]).square().mean())
        log.append({'epoch':epoch+1,'train_MSE':total/n,'validation_MSE':val})
        if val<best: best=val;best_weights=copy.deepcopy(model.state_dict())
        recovery.save(epoch+1,stateful,{"best":best,"best_weights":best_weights,"log":log})
        print(f'{args.group} {args.model} epoch {epoch+1}/{epochs}: train MSE={total/n:.6f}, validation MSE={val:.6f}')
    if best_weights is None: raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_weights);model.eval()
    with torch.no_grad():pred=model(dataset.test[0]).cpu().numpy()
    actual=dataset.test[1].cpu().numpy()
    report={'model':args.model,'doi':cfg['doi'],'experiment':args.group,
            'test_cell':dataset.test_cell,'train_cells':dataset.train_cells,
            'run_type':'reduced-width pipeline check' if args.smoke else 'real NASA cycle-level experiment',
            'metrics':evaluate(actual,pred),'data':dataset.metadata,
            'paper_comparison':'New independent run; not the publication’s reported metrics'}
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(log).to_csv(args.out/'training_history.csv',index=False)
    pd.DataFrame({'cycle':dataset.test_cycles,'observed_SOH':actual,
                  'predicted_SOH':pred}).to_csv(args.out/'test_predictions.csv',index=False)
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    torch.save({'state_dict':best_weights,'model':args.model,'model_settings':settings,
                'preprocessing':dataset.metadata},args.out/'checkpoint.pt')
    print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__':
    main()
