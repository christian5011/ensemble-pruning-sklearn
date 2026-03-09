# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `ensemble_pruning/`. `ensemblepruning.py` contains the `EnsemblePruningClassifier`, and `pruning_state.py` contains pruning strategy implementations. Public exports are defined in `ensemble_pruning/__init__.py`. Example usage lives in `examples/ensemble_pruning_demo.py`. Packaging metadata is in `setup.py`, runtime dependencies are listed in `requirements.txt`, and container tooling is under `.devcontainer/` and `dev/`.

## Build, Test, and Development Commands
Install the package in editable mode with `pip install -e .`; this is the default setup for local development. Install runtime dependencies with `pip install -r requirements.txt`. Run the example script with `python examples/ensemble_pruning_demo.py` to exercise the estimator end to end and regenerate `pruning_performance_curve.png`. Build a distributable package with `python setup.py sdist bdist_wheel` when you need to verify packaging metadata.

## Coding Style & Naming Conventions
Follow standard Python style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and module names in lowercase. Keep docstrings concise and NumPy-style, matching the existing estimator API documentation. For scikit-learn compatibility, fitted attributes should use a trailing underscore, such as `classes_` or `use_n_estimators_`. Prefer small, focused helpers over deeply nested logic.

## Testing Guidelines
There is no dedicated `tests/` directory yet, so new contributions should add targeted automated tests alongside the change. Use `pytest` conventions if you introduce a suite: place tests under `tests/`, name files `test_*.py`, and cover estimator fitting, prediction, and custom pruning criteria. Until a suite exists, run `python examples/ensemble_pruning_demo.py` and a quick import check such as `python -c "from ensemble_pruning import EnsemblePruningClassifier"` before submitting.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Add custom extensions` and `Update .gitignore`. Keep commit titles concise, capitalized, and focused on one change. Pull requests should include a brief summary, note any API or dependency changes, describe how the change was validated, and attach updated output or screenshots only when the demo or generated plot changes.

## Security & Configuration Tips
Keep the package lightweight: avoid adding heavy dependencies unless they are required by the public API. Do not commit local virtual environments or generated artifacts beyond intentional example outputs. If you change dependencies, update both `setup.py` and `requirements.txt` together.
