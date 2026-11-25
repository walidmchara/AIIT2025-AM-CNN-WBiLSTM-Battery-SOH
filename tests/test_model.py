import torch
import pytest
from aiit_soh.model import morlet, make_model
from aiit_soh.metrics import evaluate

SMALL={'conv_filters':(8,12,16),'kernels':(3,3,3),'pools':(2,2,2),'lstm_hidden':8,'attention_units':12}


def test_wavelet_formula():
    x=torch.tensor([-1.,0.,1.]); expected=torch.cos(2.1*x)*torch.exp(-x*x/3)
    assert torch.allclose(morlet(x),expected)


def test_configured_wavelet_parameters_are_used():
    torch.manual_seed(7)
    m=make_model('AM-CNN-WBiLSTM',features=7,**SMALL,
                 morlet_frequency=1.2,morlet_variance_divisor=2.0)
    m.eval()
    x=torch.ones(2,16,7)
    configured=m(x)
    m.morlet_frequency=2.8
    changed=m(x)
    assert not torch.allclose(configured,changed)


@pytest.mark.parametrize('kind',['AM-CNN-WBiLSTM','AM-CNN-BiLSTM','AM-CNN-LSTM','CNN-BiLSTM','CNN-LSTM'])
def test_variants(kind):
    m=make_model(kind,features=7,**SMALL)
    out=m(torch.randn(3,16,7))
    assert out.shape==(3,)
    assert torch.isfinite(out).all()
    if 'AM-' in kind:
        assert torch.allclose(m.last_attention.sum(-1),torch.ones(3),atol=1e-5)


def test_minimum_window():
    m=make_model('AM-CNN-WBiLSTM',features=7,**SMALL)
    with pytest.raises(ValueError):m(torch.randn(2,7,7))


def test_regression_metrics():
    d=evaluate([1., .8],[.9,.7]);assert abs(d['MAE']-.1)<1e-6
    assert d['RMSE']>0 and d['MAPE_percent']>0
