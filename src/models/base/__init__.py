"""Modelos base individuais do pool (M = 8)."""

from src.models.base.amf import create_amf
from src.models.base.bagging_ht import create_bagging_ht
from src.models.base.efdt import create_efdt
from src.models.base.gaussian_nb import create_gaussian_nb
from src.models.base.hoeffding_adaptive_tree import create_hoeffding_adaptive_tree
from src.models.base.hoeffding_tree import create_hoeffding_tree
from src.models.base.knn import create_knn
from src.models.base.softmax_regression import create_softmax_regression

__all__ = [
    "create_gaussian_nb",
    "create_hoeffding_tree",
    "create_hoeffding_adaptive_tree",
    "create_efdt",
    "create_softmax_regression",
    "create_knn",
    "create_bagging_ht",
    "create_amf",
]
