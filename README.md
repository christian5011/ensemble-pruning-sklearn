# Ensemble Pruning Classifier for scikit-learn

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Ensemble Pruning Classifier is a scikit-learn compatible meta-estimator that optimizes ensemble models by selecting the most effective subset of base estimators. It intelligently prunes redundant or poorly performing estimators while maintaining or improving predictive performance.

## Why Ensemble Pruning?

- 🚀 **Reduce inference time** by using fewer estimators
- 💾 **Decrease memory footprint** of your ensemble models
- 📈 **Improve generalization** by removing harmful estimators
- 🔍 **Gain insights** into estimator contributions
- 🤖 **Fully compatible** with scikit-learn's API

## Installation

```bash
# Install locally
pip install -e .

# Or install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from ensemble_pruning import EnsemblePruningClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load and prepare data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a base ensemble
base_ensemble = RandomForestClassifier(n_estimators=50, random_state=42)
base_ensemble.fit(X_train, y_train)

# 3. Create and fit the pruning classifier
pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="uwa",          # Pruning criteria
    pruning_rate=0.5         # Keep 50% of estimators
)
pruner.fit(X_train, y_train)

# 4. Use the pruned ensemble
y_pred = pruner.predict(X_test)
y_proba = pruner.predict_proba(X_test)

# 5. Check results
print(f"Original estimators: {pruner.n_estimators_}")
print(f"Used estimators: {pruner.use_n_estimators_}")
```

## Pruning Criteria

Choose the strategy for selecting the best subset of estimators:

| Criteria | Default | Description | When to Use |
|---------|---------|-------------|-------------|
| `max_proba` | ✓ | Maximizes average class probability | General purpose, most reliable |
| `max_voted` | | Maximizes majority vote accuracy | When probability calibration matters |
| `complement` | | Prioritizes estimators that correct ensemble errors | For imbalanced datasets |
| `uwa` | | Uses Uncertainty Weighted Accuracy metric | Complex decision boundaries |
| Custom Object | | Implement your own `PruningState` subclass | Specialized requirements |

## Advanced Features

### Automatic Optimal Pruning

Automatically selects the optimal number of estimators without specifying `pruning_rate`:

```python
pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="uwa"
)

pruner.fit(X_train, y_train)
print(f"Optimal estimators: {pruner.use_n_estimators_}")
```

### Evaluate Pruning Performance

Check error rates for different ensemble sizes and visualize the performance curve:

```python
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Check error rates for different ensemble sizes
errors = pruner.check_error_performance(X_test, y_test)

# Plot performance curve
n_estimators = range(1, pruner.n_estimators_ + 1)
plt.plot(n_estimators, errors, 'b-', linewidth=2)
plt.axvline(x=pruner.use_n_estimators_, color='r', linestyle='--', 
            label=f'Optimal: {pruner.use_n_estimators_} estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Error Rate')
plt.title('Ensemble Pruning Performance Curve')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('pruning_curve.png', dpi=300)
```

### Predict with Custom Estimator Count

Predict using a specific number of estimators (useful for early stopping or analysis):

```python
# Predict using only 10 estimators
y_pred_10 = pruner.predict_n_estims(X_test, n_estims=10)

# Get probabilities with 20 estimators
y_proba_20 = pruner.predict_proba_n_estims(X_test, n_estims=20)

# Accuracy comparison
print(f"Using all {pruner.use_n_estimators_} estimators:")
print(f"  Train: {accuracy_score(y_train, pruner.predict(X_train)):.4f}")
print(f"  Test:  {accuracy_score(y_test, pruner.predict(X_test)):.4f}")

print(f"\nUsing only 10 estimators:")
print(f"  Train: {accuracy_score(y_train, pruner.predict_n_estims(X_train, n_estims=10)):.4f}")
print(f"  Test:  {accuracy_score(y_test, pruner.predict_n_estims(X_test, n_estims=10)):.4f}")
```

### Custom Pruning Strategy

Implement your own pruning strategy by subclassing `PruningState`:

```python
import numpy as np
from ensemble_pruning.pruning_state import PruningState

class MyCustomStrategy(PruningState):
    def start(self, ensemble_pruning, X, y):
        # Initialize state
        self.state_ = np.zeros((self.n_samples_, self.n_classes_))
        # Add custom initialization here
        return
    
    def update(self, idx, score):
        # Update state with new estimator
        # Add custom update logic here
        return
    
    def partial_score(self, idx):
        # Calculate score for candidate estimator
        # Return your custom metric
        return custom_score

pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria=MyCustomStrategy()
)
```

## Full Example with Performance Comparison

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Generate custom dataset
X = np.random.randn(1000, 10)
y = (np.random.randn(1000) > 0).astype(int)

# Create base ensemble
base_ensemble = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train base ensemble
base_ensemble.fit(X, y)
base_train_acc = base_ensemble.score(X, y)
base_test_acc = base_ensemble.score(X, y)
# Using same data for demo

# Apply ensemble pruning
pruner = EnsemblePruningClassifier(
    base_ensemble=base_ensemble,
    criteria="uwa"
)

pruner.fit(X, y)
pruned_train_acc = pruner.score(X, y)
pruned_test_acc = pruner.score(X, y)

print(f"Base ensemble:")
print(f"  Accuracy: {base_train_acc:.4f}")
print(f"  Estimators: {pruner.n_estimators_}")

print(f"\nPruned ensemble:")
print(f"  Accuracy: {pruned_train_acc:.4f}")
print(f"  Estimators: {pruner.use_n_estimators_} ({(100 - pruner.use_n_estimators_ * 100 / pruner.n_estimators_):.0f}% reduction)")

# Show classification report
print("\nClassification Report:")
print(classification_report(y, pruner.predict(X)))
```

## How It Works

1. **Initialization**: Takes a pre-trained ensemble with `estimators_` attribute
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_ensemble` | estimator object | required | Pre-trained ensemble with `estimators_` attribute |
| `criteria` | str or PruningState | "max_proba" | Pruning strategy |
| `pruning_rate` | float or None | None | Fraction of estimators to keep. If None, auto-selects optimal number |

### Key Attributes After Fitting

| Attribute | Description |
|-----------|-------------|
| `use_n_estimators_` | Number of estimators actually used for prediction |
| `ordered_idx_` | Indices of base estimators sorted by contribution (best first) |
| `n_estimators_` | Total number of base estimators |
| `estimators_` | Reference to base ensemble's estimators |

## Performance Considerations

- ✅ **Works with any scikit-learn ensemble** that has `estimators_` attribute
- ✅ **Faster inference** proportional to pruning rate (50% pruning ≈ 2x speedup)
- ✅ **No retraining required** - uses pre-trained models
- ⚠️ **Requires pre-trained ensemble** - must provide trained `base_ensemble`
- ⚠️ **Estimator count selection** - validate on separate set if using auto-selection
- 📊 **Best practices** - Use validation set for determining optimal `pruning_rate`

## Supported Algorithms

Works with any scikit-learn classifier compatible with:

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `ExtraTreesClassifier`
- `VotingClassifier`
- Any custom ensemble with `estimators_` and `predict`/`predict_proba` methods

## License

Distributed under the BSD 3-Clause License. See `LICENSE` for details.

```text
Copyright (c) 2018, Christian Messina Valverde.
All rights reserved.
```

---

**Contributions welcome!** Please open issues for bug reports or feature requests, and submit pull requests for improvements.
