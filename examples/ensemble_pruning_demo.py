#!/usr/bin/env python3

"""Benchmark-style demo for EnsemblePruningClassifier.

This script runs ensemble pruning on several built-in scikit-learn datasets,
compares the supported pruning criteria against the unpruned base ensemble,
and saves charts with the results.

Usage
-----
python examples/ensemble_pruning_demo.py

Outputs
-------
- demo_outputs/accuracy_by_dataset.png
- demo_outputs/estimator_usage_by_dataset.png
- demo_outputs/performance_curves.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ensemble_pruning import EnsemblePruningClassifier


CRITERIA = ("max_proba", "max_voted", "complement", "uwa")


@dataclass
class DatasetConfig:
    """Configuration for a single benchmark dataset."""

    name: str
    loader: Callable


@dataclass
class CriterionResult:
    """Evaluation summary for one pruning criterion."""

    criterion: str
    train_accuracy: float
    test_accuracy: float
    selected_estimators: int
    total_estimators: int
    error_curve: list

    @property
    def usage_ratio(self):
        return self.selected_estimators / float(self.total_estimators)

    @property
    def reduction_ratio(self):
        return 1.0 - self.usage_ratio


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="demo_outputs",
        help="Directory where the demo charts will be written.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="Number of trees in the base RandomForestClassifier.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split ratio for each dataset.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split and the forest.",
    )
    return parser.parse_args()


def dataset_configs():
    """Return the built-in scikit-learn datasets used in the demo."""
    return [
        DatasetConfig("Iris", load_iris),
        DatasetConfig("Wine", load_wine),
        DatasetConfig("Breast Cancer", load_breast_cancer),
        DatasetConfig("Digits", load_digits),
    ]


def train_base_ensemble(X_train, y_train, n_estimators, random_state):
    """Fit and return the reference RandomForestClassifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_dataset(config, n_estimators, test_size, random_state):
    """Run the pruning benchmark for one dataset."""
    bunch = config.loader()
    X, y = bunch.data, bunch.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    base_ensemble = train_base_ensemble(
        X_train,
        y_train,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    base_result = {
        "train_accuracy": accuracy_score(y_train, base_ensemble.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, base_ensemble.predict(X_test)),
        "n_estimators": base_ensemble.n_estimators,
    }

    criterion_results = []
    for criterion in CRITERIA:
        pruner = EnsemblePruningClassifier(
            base_ensemble=base_ensemble,
            criteria=criterion,
        )
        pruner.fit(X_train, y_train)

        criterion_results.append(
            CriterionResult(
                criterion=criterion,
                train_accuracy=accuracy_score(y_train, pruner.predict(X_train)),
                test_accuracy=accuracy_score(y_test, pruner.predict(X_test)),
                selected_estimators=pruner.use_n_estimators_,
                total_estimators=pruner.n_estimators_,
                error_curve=pruner.check_error_performance(X_test, y_test),
            )
        )

    return {
        "config": config,
        "bunch": bunch,
        "X_shape": X.shape,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "base_result": base_result,
        "criterion_results": criterion_results,
    }


def print_dataset_summary(result):
    """Print a readable summary for one dataset."""
    config = result["config"]
    base = result["base_result"]

    print("=" * 72)
    print(config.name.upper())
    print("=" * 72)
    print(
        "Samples: {samples} | Features: {features} | Train: {train} | Test: {test}".format(
            samples=result["X_shape"][0],
            features=result["X_shape"][1],
            train=result["train_size"],
            test=result["test_size"],
        )
    )
    print(
        "Base ensemble -> estimators: {estimators}, train acc: {train:.4f}, test acc: {test:.4f}".format(
            estimators=base["n_estimators"],
            train=base["train_accuracy"],
            test=base["test_accuracy"],
        )
    )
    print("-" * 72)
    print(
        "{:<12} {:>12} {:>12} {:>12} {:>12}".format(
            "Criterion",
            "Train Acc",
            "Test Acc",
            "Used Est.",
            "Reduction",
        )
    )
    for item in result["criterion_results"]:
        print(
            "{:<12} {:>12.4f} {:>12.4f} {:>12d} {:>11.1%}".format(
                item.criterion,
                item.train_accuracy,
                item.test_accuracy,
                item.selected_estimators,
                item.reduction_ratio,
            )
        )
    print()


def plot_accuracy_summary(results, output_dir):
    """Plot base vs pruned test accuracy for each dataset."""
    dataset_names = [result["config"].name for result in results]
    x = np.arange(len(dataset_names))
    series_labels = ["base"] + list(CRITERIA)
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6))

    base_scores = [result["base_result"]["test_accuracy"] for result in results]
    ax.bar(x - 2 * width, base_scores, width=width, label="base", color="#34495e")

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, criterion in enumerate(CRITERIA):
        scores = []
        for result in results:
            item = next(
                current
                for current in result["criterion_results"]
                if current.criterion == criterion
            )
            scores.append(item.test_accuracy)
        ax.bar(
            x + (idx - 1) * width,
            scores,
            width=width,
            label=criterion,
            color=colors[idx],
        )

    ax.set_title("Test Accuracy by Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=len(series_labels), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_usage_summary(results, output_dir):
    """Plot the fraction of estimators kept by each pruning criterion."""
    dataset_names = [result["config"].name for result in results]
    x = np.arange(len(dataset_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, criterion in enumerate(CRITERIA):
        ratios = []
        for result in results:
            item = next(
                current
                for current in result["criterion_results"]
                if current.criterion == criterion
            )
            ratios.append(item.usage_ratio)
        ax.bar(
            x + (idx - 1.5) * width,
            ratios,
            width=width,
            label=criterion,
            color=colors[idx],
        )

    ax.set_title("Fraction of Estimators Retained")
    ax.set_ylabel("Selected / Total Estimators")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "estimator_usage_by_dataset.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_performance_curves(results, output_dir):
    """Plot test error curves for each criterion on every dataset."""
    n_results = len(results)
    ncols = 2
    nrows = int(np.ceil(n_results / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = {
        "max_proba": "#1f77b4",
        "max_voted": "#2ca02c",
        "complement": "#ff7f0e",
        "uwa": "#d62728",
    }

    for axis, result in zip(axes, results):
        for item in result["criterion_results"]:
            axis.plot(
                range(1, item.total_estimators + 1),
                item.error_curve,
                linewidth=2,
                label=item.criterion,
                color=colors[item.criterion],
            )
            axis.axvline(
                item.selected_estimators,
                color=colors[item.criterion],
                linestyle="--",
                alpha=0.3,
            )

        axis.set_title(result["config"].name)
        axis.set_xlabel("Number of Estimators")
        axis.set_ylabel("Test Error")
        axis.grid(True, linestyle="--", alpha=0.4)

    for axis in axes[n_results:]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(CRITERIA))
    fig.suptitle("Pruning Performance Curves Across Datasets", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "performance_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the multi-dataset pruning demo."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ENSEMBLE PRUNING MULTI-DATASET DEMO")
    print("=" * 72)
    print("Datasets:", ", ".join(config.name for config in dataset_configs()))
    print("Base estimators per forest:", args.n_estimators)
    print("Output directory:", output_dir)
    print()

    results = []
    for config in dataset_configs():
        result = evaluate_dataset(
            config,
            n_estimators=args.n_estimators,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        results.append(result)
        print_dataset_summary(result)

    plot_accuracy_summary(results, output_dir)
    plot_usage_summary(results, output_dir)
    plot_performance_curves(results, output_dir)

    print("=" * 72)
    print("Charts written to:")
    print("  -", output_dir / "accuracy_by_dataset.png")
    print("  -", output_dir / "estimator_usage_by_dataset.png")
    print("  -", output_dir / "performance_curves.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
