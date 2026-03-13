# Ensemble Pruning for scikit-learn

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

`ensemble_pruning` is a small scikit-learn compatible meta-estimator for pruning an already-fitted ensemble classifier down to a stronger or smaller subset of base estimators.

The project centers on `EnsemblePruningClassifier`, which:

- reorders base estimators by their contribution to the ensemble,
- keeps either a fixed fraction or the empirically best prefix,
- preserves a familiar `fit` / `predict` / `predict_proba` workflow.

## What It Does

Ensemble pruning can help when a fitted ensemble is larger than necessary. Instead of retraining a new model, this package evaluates the base estimators already present in the ensemble and keeps only the most useful ones for prediction.

Typical goals include:

- reducing inference cost,
- shrinking memory usage,
- removing weak or redundant estimators,
- experimenting with different pruning criteria.

## Installation

Install the package in editable mode for local development:

```bash
pip install -e .
```

Install the example and local development extras as well:

```bash
pip install -r requirements-dev.txt
```

If your environment uses `python3` rather than `python`, use the matching `pip3` / `python3` commands.

To mirror the devcontainer setup locally with a virtual environment:

```bash
bash dev/setup_local_env.sh
```

By default the script creates `.venv/` in the repository root. You can pass a custom path if you prefer:

```bash
bash dev/setup_local_env.sh /path/to/venv
```

## Requirements and Compatibility

Package requirements in `setup.py`:

- `numpy>=1.15.0`
- `scikit-learn>=0.20.0`

The demo script also uses `matplotlib`, which is listed in `requirements-dev.txt`.

`EnsemblePruningClassifier` expects a classifier ensemble that is already fitted and exposes:

- `estimators_`
- `classes_`
- base estimators with `predict_proba`

In practice, this package is best suited to fitted classifier ensembles whose members support class probabilities.

## Quick Start

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ensemble_pruning import EnsemblePruningClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

base_ensemble = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
)
base_ensemble.fit(X_train, y_train)

pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="uwa",
    pruning_rate=0.5,
)
pruner.fit(X_train, y_train)

print("Total estimators:", pruner.n_estimators_)
print("Estimators used:", pruner.use_n_estimators_)
print("Accuracy:", pruner.score(X_test, y_test))
```

## Pruning Criteria

The `criteria` argument accepts either a built-in strategy name or a custom `PruningState` instance.

Built-in options:

- `max_proba`: greedily maximizes predictive performance using averaged class probabilities.
- `max_voted`: greedily maximizes majority-vote performance.
- `complement`: favors estimators that correct mistakes made by the current sub-ensemble.
- `uwa`: uses the Uncertainty Weighted Accuracy criterion.

## Automatic vs Fixed Pruning

Use a fixed pruning rate when you already know how aggressively you want to reduce the ensemble:

```python
pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="max_proba",
    pruning_rate=0.5,
)
```

Leave `pruning_rate=None` to let the estimator choose the best prefix length during `fit`:

```python
pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="max_proba",
)
```

After fitting:

- `n_estimators_` is the total size of the original ensemble.
- `use_n_estimators_` is the number of estimators selected for prediction.
- `ordered_idx_` is the ranked order of base estimator indices.

## Prediction Helpers

The estimator supports prediction with either the selected subset or a user-chosen prefix of the ranked ensemble:

```python
y_pred = pruner.predict(X_test)
y_proba = pruner.predict_proba(X_test)

y_pred_10 = pruner.predict_n_estims(X_test, n_estims=10)
y_proba_10 = pruner.predict_proba_n_estims(X_test, n_estims=10)
```

You can also inspect performance across prefix lengths:

```python
errors = pruner.check_error_performance(X_test, y_test)
```

## Custom Pruning Strategies

Custom strategies are implemented by subclassing `PruningState` and passing an instance through `criteria`.

```python
import numpy as np

from ensemble_pruning import EnsemblePruningClassifier, PruningState


class MyStrategy(PruningState):
    def start(self, ensemble_pruning, X, y):
        super().start(ensemble_pruning, X, y)
        self.state_ = np.zeros((self.n_samples_, self.n_classes_))

    def update(self, idx, score):
        super().update(idx, score)

    def partial_score(self, idx):
        return 0.0


pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria=MyStrategy(),
)
```

See [ensemble_pruning/pruning_state.py](/home/christian/repos/ensemble-pruning-sklearn/ensemble_pruning/pruning_state.py) for the built-in strategy implementations.

## Example Script

The repository includes a runnable demo:

```bash
python examples/ensemble_pruning_demo.py
```

If needed in your environment:

```bash
python3 examples/ensemble_pruning_demo.py
```

The demo:

- runs on multiple built-in scikit-learn datasets,
- compares the supported pruning criteria against the base ensemble,
- prints a summary table for each dataset,
- saves charts under `demo_outputs/`.

## Project Layout

- [ensemble_pruning/ensemblepruning.py](/home/christian/repos/ensemble-pruning-sklearn/ensemble_pruning/ensemblepruning.py): `EnsemblePruningClassifier`
- [ensemble_pruning/pruning_state.py](/home/christian/repos/ensemble-pruning-sklearn/ensemble_pruning/pruning_state.py): pruning strategy implementations
- [ensemble_pruning/__init__.py](/home/christian/repos/ensemble-pruning-sklearn/ensemble_pruning/__init__.py): public exports and version
- [examples/ensemble_pruning_demo.py](/home/christian/repos/ensemble-pruning-sklearn/examples/ensemble_pruning_demo.py): end-to-end example
- [setup.py](/home/christian/repos/ensemble-pruning-sklearn/setup.py): package metadata
- [dev/setup_local_env.sh](/home/christian/repos/ensemble-pruning-sklearn/dev/setup_local_env.sh): local virtual environment bootstrap script
- [requirements-dev.txt](/home/christian/repos/ensemble-pruning-sklearn/requirements-dev.txt): local development and demo dependencies

## Development Notes

Build a source distribution and wheel with:

```bash
python setup.py sdist bdist_wheel
```

There is not yet a dedicated `tests/` directory in this repository. For now, a reasonable smoke-check is:

```bash
python -c "from ensemble_pruning import EnsemblePruningClassifier"
python examples/ensemble_pruning_demo.py
```

## Limitations

- The base ensemble must already be fitted before you construct `EnsemblePruningClassifier`.
- The current implementation is for classification workflows.
- Probability-based criteria require base estimators that implement `predict_proba`.
- Automatic pruning chooses the best prefix on the data passed to `fit`, so a separate validation split is still the safer option for model selection.

## License

BSD 3-Clause.
