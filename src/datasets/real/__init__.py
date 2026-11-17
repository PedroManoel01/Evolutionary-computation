"""Bases reais disponíveis diretamente no river."""

from src.datasets.real.credit_card import create_credit_card
from src.datasets.real.elec2 import create_elec2

__all__ = [
    "create_elec2",
    "create_credit_card",
]
