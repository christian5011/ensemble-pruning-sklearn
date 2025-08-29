"""Ensemble pruning meta-estimator for scikit-learn."""

# Author: Christian Messina <christian.messina.val@gmail.com>
# License: BSD-3-Clause

__version__ = '0.1.0'

from .ensemblepruning import EnsemblePruningClassifier
from .pruning_state import PruningState

__all__ = [
    'EnsemblePruningClassifier',
    'PruningState',
    '__version__'
]