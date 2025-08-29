#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Pruning Demo

This script demonstrates how to use the EnsemblePruningClassifier with a RandomForest.
It shows basic usage, performance comparison, and advanced features of the pruning technique.

Usage:
    python ensemble_pruning_demo.py

Requirements:
    numpy, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ensemble_pruning import EnsemblePruningClassifier


def main():
    """Main function demonstrating EnsemblePruningClassifier usage."""
    print("="*60)
    print("ENSEMBLE PRUNING CLASSIFIER DEMO")
    print("="*60)
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing dataset...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"  - Dataset: Iris (n_samples={X.shape[0]}, n_features={X.shape[1]})")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")

    # 2. Train base ensemble
    print("\n2. Training base ensemble (RandomForestClassifier)...")
    base_ensemble = RandomForestClassifier(
        n_estimators=50, 
        random_state=42,
        n_jobs=-1
    )
    base_ensemble.fit(X_train, y_train)
    
    # Evaluate base ensemble
    base_train_acc = accuracy_score(y_train, base_ensemble.predict(X_train))
    base_test_acc = accuracy_score(y_test, base_ensemble.predict(X_test))
    print(f"  - Base ensemble trained with {base_ensemble.n_estimators} trees")
    print(f"  - Base ensemble training accuracy: {base_train_acc:.4f}")
    print(f"  - Base ensemble test accuracy: {base_test_acc:.4f}")

    # 3. Apply ensemble pruning
    print("\n3. Applying ensemble pruning...")
    
    # Option 1: Pruning with fixed rate (50%)
    print("\n  Option 1: Pruning with fixed rate (50%)")
    pruner_fixed = EnsemblePruningClassifier(
        base_ensemble=base_ensemble,
        criteria="uwa",
        pruning_rate=0.5
    )
    pruner_fixed.fit(X_train, y_train)
    
    fixed_train_acc = accuracy_score(y_train, pruner_fixed.predict(X_train))
    fixed_test_acc = accuracy_score(y_test, pruner_fixed.predict(X_test))
    
    print(f"  - Pruned ensemble uses {pruner_fixed.use_n_estimators_} out of {pruner_fixed.n_estimators_} estimators")
    print(f"  - Pruned ensemble training accuracy: {fixed_train_acc:.4f}")
    print(f"  - Pruned ensemble test accuracy: {fixed_test_acc:.4f}")
    print(f"  - Size reduction: {(1 - pruner_fixed.use_n_estimators_/pruner_fixed.n_estimators_)*100:.1f}%")
    print(f"  - Speed impact: ~{pruner_fixed.use_n_estimators_/pruner_fixed.n_estimators_*100:.1f}% of original inference time")

    # Option 2: Automatic optimal pruning
    print("\n  Option 2: Automatic optimal pruning (no pruning rate specified)")
    pruner_auto = EnsemblePruningClassifier(
        base_ensemble=base_ensemble,
        criteria="uwa"
    )
    pruner_auto.fit(X_train, y_train)
    
    auto_train_acc = accuracy_score(y_train, pruner_auto.predict(X_train))
    auto_test_acc = accuracy_score(y_test, pruner_auto.predict(X_test))
    
    print(f"  - Automatically selected {pruner_auto.use_n_estimators_} out of {pruner_auto.n_estimators_} estimators")
    print(f"  - Pruned ensemble training accuracy: {auto_train_acc:.4f}")
    print(f"  - Pruned ensemble test accuracy: {auto_test_acc:.4f}")
    print(f"  - Size reduction: {(1 - pruner_auto.use_n_estimators_/pruner_auto.n_estimators_)*100:.1f}%")
    print(f"  - Speed impact: ~{pruner_auto.use_n_estimators_/pruner_auto.n_estimators_*100:.1f}% of original inference time")

    # 4. Compare different pruning criteria
    print("\n4. Comparing different pruning criteria...")
    criteria_list = ["max_proba", "max_voted", "complement", "uwa"]
    results = []
    
    for criteria in criteria_list:
        pruner = EnsemblePruningClassifier(
            base_ensemble=base_ensemble,
            criteria=criteria
        )
        pruner.fit(X_train, y_train)
        
        test_acc = accuracy_score(y_test, pruner.predict(X_test))
        size_ratio = pruner.use_n_estimators_ / pruner.n_estimators_
        
        results.append((criteria, test_acc, pruner.use_n_estimators_))
        print(f"  - Criteria '{criteria}': {pruner.use_n_estimators_} estimators, test accuracy = {test_acc:.4f}")
    
    # 5. Analyze pruning performance
    print("\n5. Analyzing pruning performance curve...")
    errors = pruner_auto.check_error_performance(X_test, y_test)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, pruner_auto.n_estimators_ + 1), errors, 'b-', linewidth=2)
    plt.axvline(x=pruner_auto.use_n_estimators_, color='r', linestyle='--', 
                label=f'Optimal: {pruner_auto.use_n_estimators_} estimators')
    plt.xlabel('Number of Estimators', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Ensemble Pruning Performance Curve', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('pruning_performance_curve.png', dpi=300, bbox_inches='tight')
    print("  - Performance curve saved as 'pruning_performance_curve.png'")
    
    # 6. Advanced usage: Predict with custom number of estimators
    print("\n6. Advanced usage: Predict with custom number of estimators...")
    n_estims = max(1, pruner_auto.use_n_estimators_ // 2)
    y_pred_custom = pruner_auto.predict_n_estims(X_test, n_estims=n_estims)
    custom_acc = accuracy_score(y_test, y_pred_custom)
    
    print(f"  - Predicting with only {n_estims} estimators")
    print(f"  - Accuracy with {n_estims} estimators: {custom_acc:.4f}")
    
    # Show classification report for the automatically pruned model
    print("\n7. Detailed classification report for automatically pruned model:")
    print(classification_report(y_test, pruner_auto.predict(X_test), 
                               target_names=load_iris().target_names))

    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()