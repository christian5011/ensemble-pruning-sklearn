# Ensemble Pruning Classifier for scikit-learn

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Ensemble Pruning Classifier is a scikit-learn compatible meta-estimator that optimizes ensemble models by selecting the most effective subset of base estimators. It intelligently prunes redundant or poorly performing estimators while maintaining or improving predictive performance.

## Why Ensemble Pruning?

- 🚀 **Reduce inference time** by using fewer estimators
- 💾 **Decrease memory footprint** of your ensemble models
- 📈 **Improve generalization** by removing harmful estimators
- 🔍 **Gain insights** into estimator contributions
- 🤖 **Fully compatible** with scikit-learn's API


## Basic Usage

```python
from sklearn.ensemble import RandomForestClassifier
from ensemble_pruning import EnsemblePruningClassifier

# 1. Train a base ensemble (e.g., Random Forest)
base_ensemble = RandomForestClassifier(n_estimators=100, random_state=42)
base_ensemble.fit(X_train, y_train)

# 2. Create and fit the pruning classifier
pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="uwa",          # Pruning criteria (see below)
    pruning_rate=0.5         # Keep 50% of estimators
)
pruner.fit(X_train, y_train)

# 3. Use the pruned ensemble for predictions
y_pred = pruner.predict(X_test)
y_proba = pruner.predict_proba(X_test)

# 4. Check how many estimators were used
print(f"Original estimators: {pruner.n_estimators_}")
print(f"Used estimators: {pruner.use_n_estimators_}")
```

## Pruning Criteria

Choose the strategy for selecting the best subset of estimators:

| Criteria       | Description                                                                 | When to Use                          |
|----------------|-----------------------------------------------------------------------------|--------------------------------------|
| `max_proba`    | Maximizes average class probability (default)                               | General purpose, most reliable       |
| `max_voted`    | Maximizes majority vote accuracy                                            | When probability calibration matters |
| `complement`   | Prioritizes estimators that correct ensemble errors                         | For imbalanced datasets              |
| `uwa`          | Uses Uncertainty Weighted Accuracy metric                                   | Complex decision boundaries          |
| Custom Object  | Implement your own `PruningState` subclass                                  | Specialized requirements             |

## Advanced Features

### Automatic Optimal Pruning (without pruning rate)
```python
pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="uwa"
)
pruner.fit(X_train, y_train)
# Automatically selects optimal number of estimators
```

### Evaluate Pruning Performance
```python
# Check error rates for different ensemble sizes
errors = pruner.check_error_performance(X_val, y_val)

import matplotlib.pyplot as plt
plt.plot(range(1, pruner.n_estimators_ + 1), errors)
plt.xlabel('Number of Estimators')
plt.ylabel('Error Rate')
plt.title('Pruning Performance Curve')
plt.show()
```

### Predict with Custom Estimator Count
```python
# Predict using only 10 estimators
y_pred = pruner.predict_n_estims(X_test, n_estims=10)

# Get probabilities with 20 estimators
y_proba = pruner.predict_proba_n_estims(X_test, n_estims=20)
```

### Custom Pruning Strategy
```python
from ensemble_pruning.pruning_state import PruningState

class MyCustomStrategy(PruningState):
    def start(self, ensemble_pruning, X, y):
        # Initialize state
        pass
        
    def update(self, idx, score):
        # Update state with new estimator
        pass
        
    def partial_score(self, idx):
        # Calculate score for candidate estimator
        return custom_score

pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria=MyCustomStrategy()
)
```

## How It Works

1. **Initialization**: Takes a pre-trained ensemble (e.g., RandomForest)
2. **Reordering**: 
   - Evaluates each estimator's contribution to the current sub-ensemble
   - Sorts estimators by their individual contribution score
3. **Pruning**:
   - Selects top *k* estimators (where *k* = `n_estimators * pruning_rate`)
   - If no pruning rate is specified, selects *k* that maximizes validation accuracy
4. **Prediction**:
   - Uses only the selected subset for final predictions
   - Maintains scikit-learn's standard prediction interface

## API Reference

### `EnsemblePruningClassifier` Parameters

| Parameter        | Type                | Default     | Description                                                                 |
|------------------|---------------------|-------------|-----------------------------------------------------------------------------|
| `base_ensemble`  | estimator object    | **required**| Pre-trained ensemble with `estimators_` attribute                           |
| `criteria`       | str or PruningState | `"max_proba"`| Pruning strategy (`"max_proba"`, `"max_voted"`, `"complement"`, `"uwa"`)    |
| `pruning_rate`   | float or None       | `None`      | Fraction of estimators to keep (if `None`, auto-selects optimal number)     |

### Key Attributes After Fitting

| Attribute             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `use_n_estimators_`   | Number of estimators actually used for prediction                           |
| `ordered_idx_`        | Indices of base estimators sorted by contribution (best first)              |
| `n_estimators_`       | Total number of base estimators                                             |
| `estimators_`         | Reference to base ensemble's estimators                                     |

## Performance Considerations

- ✅ **Works with any scikit-learn ensemble** that has `estimators_` attribute
- ⚠️ **Requires pre-trained ensemble** (doesn't train base models)
- ⚡ **Faster inference** proportional to pruning rate (50% pruning ≈ 2x speedup)
- 📊 **Validation recommended**: Use separate validation set for pruning decisions

## License

Distributed under the [BSD 3-Clause License](LICENSE). See `LICENSE` for details.

```
Copyright (c) 2018, Christian Messina Valverde.
All rights reserved.
```

---

**Contributions welcome!** Please open issues for bug reports or feature requests, and submit pull requests for improvements.