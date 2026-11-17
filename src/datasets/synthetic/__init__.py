"""Streams sintéticas geradas via river.datasets.synth."""

from src.datasets.synthetic.agr_abr import create_agr_abr
from src.datasets.synthetic.agr_grad import create_agr_grad
from src.datasets.synthetic.rbf_inc import create_rbf_inc
from src.datasets.synthetic.sea_rec import create_sea_rec
from src.datasets.synthetic.stagger_rec import create_stagger_rec

__all__ = [
    "create_sea_rec",
    "create_stagger_rec",
    "create_agr_abr",
    "create_agr_grad",
    "create_rbf_inc",
]
