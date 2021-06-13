"""Ensemble pruning meta-estimator."""

# Author: Christian Messina <christian.messina.val@gmail.com>
# License:

from __future__ import division

import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import accuracy_score

from ensemble_pruning.pruning_state import PruningState


__all__ = ["EnsemblePruningClassifier"]

MAX_INT = np.iinfo(np.int32).max


def _pred_max_voted(pred_matrix, idx_estimators=None):
    """Predict the max voted class"""
    if idx_estimators:  # If idx passed take the selected ones
        y_pred_aux = np.take(pred_matrix, idx_estimators, axis=0)  # (n_estimators, n_samples)
    else:  # Else take all the estimator predictions
        y_pred_aux = pred_matrix

    n_samples = pred_matrix.shape[1]

    y_pred = np.empty(n_samples)  # (n_samples)

    # Take the max voted per sample
    for i in range(n_samples):
        frecuency = y_pred_aux[:, i].astype(int)
        y_pred[i] = np.bincount(frecuency).argmax()

    return y_pred


def _pred_max_proba(proba_matrix, classes, idx_estimators=None):
    """Predict the class with max probability."""
    if idx_estimators:  # If idx passed take the selected ones
        y_proba = np.take(proba_matrix, idx_estimators, axis=0)  # (n_estimators, n_samples, n_classes)
    else:  # Else take all the probas
        y_proba = proba_matrix

    # Avg the probability per estimator
    y_proba_avg = np.mean(y_proba, axis=0)  # (n_samples, n_classes)

    # Predict the class with the max avg probability
    y_pred = classes.take((np.argmax(y_proba_avg, axis=1)), axis=0)  # (n_samples)

    return y_pred


class EnsemblePruningClassifier(BaseEstimator, ClassifierMixin):
    """An Ensemble Pruning classifier.

    An Ensemble Pruning classifier is an ensemble meta-estimator that
    selects from a set of already fitted estimators coming from an ensemble,
    a subset to optimize the prediction.
    The fit process reorders of the base estimators of the ensemble
    using a given criteria. Then it selects from the reordered sequence the first n
    estimators for prediction where n is chosen based on the efficiency of
    the subensemble in the training set.

    Parameters
    ----------
    base_ensemble : object
       The base ensemble estimator to fit on random subsets of the dataset.

    criteria : string or object, optional (default="max_proba")
       The selected criteria to use when ``fit`` is performed.

       - If "max_proba", then use the maximum probability criteria.
       - If "max_voted", then use the maximum voted criteria.
       - If "complement", then use the complementary criteria.
       - If "uwa", then use the Uncertainty Weighted Accuracy (UWA) metric.
       - If object, then check the if the object is a subclass of PruningState object, and use it.

    pruning_rate : float or None, optional (default=None)
       The pruning rate to reduce the number of estimators that will be used when predicting.
       If None, then the number of estimators used when predicting will be determined in the fit process.

    Attributes
    ----------
    estimators_ : list of classifiers
       The collection of fitted sub-estimators.

    n_estimators_ : int
       The total number of fitted sub-estimators.

    classes_ : array of shape = [n_classes]
       The classes labels.

    n_classes_ : int
       The number of classes.

    criteria_ : string or object
       The name of the selected fit criteria, or the PruningState object.

    pruning_rate_ : float
       The pruning rate from the total n_estimators.

    use_n_estimators_ : int
       The number of estimators used to predict.

    ordered_idx_ : array of shape = [n_estimators]
       The list of the ordered indexes from the base ensemble.

    """
    def __init__(self,
                 base_ensemble,
                 criteria="max_proba",
                 pruning_rate=None):

        self.estimators_ = base_ensemble.estimators_
        self.n_estimators_ = len(self.estimators_)

        self.classes_ = base_ensemble.classes_
        self.n_classes_ = len(self.classes_)

        self.criteria_ = criteria
        self.pruning_rate_ = pruning_rate
        self.ordered_idx_ = list(range(self.n_estimators_))
        # self.use_n_estimators_ = None  # Commented for fit check

    def _swap_index(self, i, j):
        """Swap two indexes from the ordered_idx_ attribute."""
        aux = self.ordered_idx_[i]
        self.ordered_idx_[i] = self.ordered_idx_[j]
        self.ordered_idx_[j] = aux

    def build_pred_matrix(self, X, n_estims=None):
        """Build a prediction matrix.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        n_estims : int or None, optional (default=None)
            The number of estimators to build the prediction matrix.
            If None, then the matrix will be build with all the estimators.

        Returns
        -------
        pred_matrix : array of shape = [n_estimators, n_samples]
            The predicted classes for each estimator.
        """
        if n_estims is None:  # If not passed, predict for all estimators
            n_estims = self.n_estimators_

        n_samples = X.shape[0]

        pred_matrix = np.empty((n_estims, n_samples))

        estimators_list = self.estimators_  # Get all estimators
        for i in range(n_estims):
            pred_matrix[i] = estimators_list[self.ordered_idx_[i]].predict(X)  # Array (n_samples)

        return pred_matrix.astype(int)

    def build_prob_matrix(self, X, n_estims=None):
        """Build a probability matrix.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        n_estims : int or None, optional (default=None)
            The number of estimators to build the probability matrix.
            If None, then the matrix will be build with all the estimators.

        Returns
        -------
        prob_matrix : array of shape = [n_estimators, n_samples, n_classes]
            The probability for each class for each estimator.
        """
        if n_estims is None:  # If not passed, predict for all estimators
            n_estims = self.n_estimators_

        n_samples = X.shape[0]

        prob_matrix = np.empty((n_estims, n_samples, self.n_classes_))

        estimators_list = self.estimators_  # Get all estimators
        for i in range(n_estims):
            prob_matrix[i] = estimators_list[self.ordered_idx_[i]].predict_proba(X)  # Array (n_samples, n_classes)

        return prob_matrix

    def fit(self, X, y):
        """Fit estimator.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])

        self.n_features_ = X.shape[1]

        self.ordered_idx_ = list(range(self.n_estimators_))  # Init ordered idx list

        state = PruningState.build(self.criteria_)  # Init state
        state.start(self, X, y)  # Start state

        pruning_n_estimators = self.n_estimators_
        if self.pruning_rate_ is not None:
            pruning_n_estimators = int(self.n_estimators_ * self.pruning_rate_)

        for i in range(self.n_estimators_):  # Iterate

            if i == pruning_n_estimators:
                # Stop reordering estimators
                break

            scores = []  # Reset the scores
            for j in range(self.n_estimators_ - i):  # Iterate trough the rest of the elements
                # Get the partial score
                score = state.partial_score(self.ordered_idx_[i+j])
                scores.append(score)  # Save the score

            best_score = max(scores)  # Take the best score
            best = scores.index(best_score)  # Take the best score index

            self._swap_index(i, best+i)  # Swap the next index with the best estimator
            state.update(self.ordered_idx_[i], best_score)  # Update state

        # Set the use n estimators for prediction
        if self.pruning_rate_ is None:
            # Set the use n estimators based on accuracy
            self.use_n_estimators_ = state.get_best_n_estimators()
        else:
            # Set the use n estimators for the selected pruning rate
            self.use_n_estimators_ = pruning_n_estimators

        return self

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        return self.predict_n_estims(X)

    def predict_n_estims(self, X, n_estims=None):
        """Predict class for X, with the indicated number of estimators.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        n_estims : int, optional (default=None)
            The number of estimators used to predict.
            If None, then it predicts with all the estimators.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        predicted_proba = self.predict_proba_n_estims(X, n_estims)
        return self.classes_.take((np.argmax(predicted_proba, axis=1)), axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        return self.predict_proba_n_estims(X)

    def predict_proba_n_estims(self, X, n_estims=None):
        """Predict class probabilities for X, with the indicated number of estimators.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        n_estims : int, optional (default=None)
            The number of estimators used to predict.
            If None, then it predicts with all the estimators.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        check_is_fitted(self, "use_n_estimators_")

        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        if n_estims is None:  # If n_estims not passed use the internal selected value
            n_estims = self.use_n_estimators_
        elif n_estims < 1 or n_estims > self.n_estimators_:
            raise ValueError("n_estims wrong value, use [1, {}]".format(self.n_estimators_))

        prob_matrix = self.build_prob_matrix(X, n_estims=n_estims)  # (n_estims, n_samples, n_classes)
        # Reduce
        y_proba = np.mean(prob_matrix, axis=0)  # (n_samples, n_classes)

        return y_proba

    def check_error_performance(self, X, y):
        """Check the prediction error for each subset of estimators.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.

        y : array-like, shape = [n_samples]
            The target class labels.

        Returns
        -------
        errors : array of shape = [n_estimators]
            The prediction error for each number of used estimators.
        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])

        errors = []
        prob_matrix = self.build_prob_matrix(X)

        for i in range(self.n_estimators_):  # Iterate
            y_pred = _pred_max_proba(prob_matrix, self.classes_, list(range(i+1)))
            score = accuracy_score(y, y_pred)  # Test the arranged estimators with the new estimator
            errors.append(1 - score)  # Save the error

        return errors
