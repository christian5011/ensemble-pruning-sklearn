"""Ensemble pruning meta-estimator."""

# Author: Christian Messina <christian.messina.val@gmail.com>
# License:

from __future__ import division

import numpy as np
import math
from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing

from timeit import default_timer as timer


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

    best_n_estimators_ : int
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
        # self.best_n_estimators_ = None  # Commented for fit check

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

        for i in range(self.n_estimators_):  # Iterate

            scores = []  # Reset the scores
            for j in range(self.n_estimators_ - i):  # Iterate trough the rest of the elements
                # Get the partial score
                score = state.partial_score(self.ordered_idx_[i+j])
                scores.append(score)  # Save the score

            best_score = max(scores)  # Take the best score
            best = scores.index(best_score)  # Take the best score index

            self._swap_index(i, best+i)  # Swap the next index with the best estimator
            state.update(self.ordered_idx_[i], best_score)  # Update state

        # Set the best n estimators for prediction
        if self.pruning_rate_ is None:  # auto
            self.best_n_estimators_ = state.get_best_n_estimators()
        else:  # manual
            self.best_n_estimators_ = int(self.n_estimators_ * self.pruning_rate_)

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
        check_is_fitted(self, "best_n_estimators_")

        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        if n_estims is None:  # If n_estims not passed use the best
            n_estims = self.best_n_estimators_
        elif n_estims < 1 or n_estims > self.n_estimators_:
            raise ValueError("n_estims wrong value, use [1-{}]".format(self.n_estimators_))

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


class PruningState(ABC):
    """Abstract class for all PruningStates.

    Warning: This class should not be used directly. Use derived classes
    instead.

     Attributes
    ----------
    n_samples_ : int
       The total number of input samples.

    classes_ : array of shape = [n_classes]
       The classes labels.

    n_classes_ : int
       The number of classes.

    y_ : array of shape = [n_samples]
       The target class labels.

    state_ : array of shape = [n_samples, n_classes]
       The accumulator state.

    scores_ : list
       The score values after each update call.
    """
    def __init__(self):
        super(PruningState, self).__init__()

        self.n_samples_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.y_ = None

        self.state_ = None
        self.scores_ = None

    @staticmethod
    def build(criteria):
        """Build and return a PruningState for the given criteria.

        Parameters
        ----------
        criteria : string or object
            The selected criteria for the PruningState.
               - If string, then use the selected state.
               - If object, then check the if the object is a subclass of PruningState object and return it.

        Returns
        -------
        state : object
            The PruningState object.
        """

        if issubclass(type(criteria), PruningState):
            state = criteria
        elif criteria == "max_voted":
            state = PruningStateMaxVoted()
        elif criteria == "max_proba":
            state = PruningStateMaxProba()
        elif criteria == "complement":
            state = PruningStateComplementary()
        elif criteria == "uwa":
            state = PruningStateUWA()
        else:  # Unknown criteria
            raise ValueError("Invalid criteria \"{}\"".format(criteria))

        return state

    @abstractmethod
    def start(self, ensemble_pruning, X, y):
        """Start the state status.

        Parameters
        ----------
        ensemble_pruning : object
            The ensemble pruning classifier.

        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target class labels.
        """
        self.n_samples_ = X.shape[0]
        self.classes_ = ensemble_pruning.classes_
        self.n_classes_ = ensemble_pruning.n_classes_
        self.y_ = y

        self.scores_ = []

        return

    @abstractmethod
    def update(self, idx, score):
        """Update the state with a new estimator.

        Parameters
        ----------
        idx : int
            The idx of the new estimator.

        score : float
            The partial score with the new estimator.
        """
        self.scores_.append(score)
        return

    @abstractmethod
    def partial_score(self, idx):
        """Get a partial score from the estimator given with the
        current sub-ensemble.

        Parameters
        ----------
        idx : int
            The estimator idx.

        Returns
        -------
        score : float
            The partial score.
        """
        return

    # Implemented methods #

    def get_best_n_estimators(self):
        """Get number of estimators that gives the best utility.

        Returns
        -------
        best_n : int
            The number of estimators from the sub-ensemble.
        """

        best_n = self.scores_.index(max(self.scores_)) + 1

        return best_n

    def pred_max_voted(self, idx, pred_matrix):
        """Predict the class with more votes, with the estimator
        given and the current sub-ensemble.

        Parameters
        ----------
        idx : int
            The estimator idx.

        pred_matrix : array of shape = [n_estimators, n_samples]
            The prediction matrix.

        Returns
        -------
        y_pred : array of shape = [n_samples]
            The predicted classes.
        """

        y_pred_aux = pred_matrix[idx]  # (n_samples)

        state_copy = self.state_.copy()  # (n_samples, n_classes)
        for i, pred in enumerate(y_pred_aux):
            state_copy[i, pred] += 1  # Add the vote of the new idx

        y_pred = np.argmax(state_copy, axis=1)  # Take the max voted class

        return y_pred

    def pred_max_proba(self, idx, proba_matrix):
        """Predict the class with more probability, with the estimator
        given and the current sub-ensemble.

        Parameters
        ----------
        idx : int
            The estimator idx.

        proba_matrix : array of shape = [n_estimators, n_samples, n_classes]
            The probability matrix.

        Returns
        -------
        y_pred : array of shape = [n_samples]
            The predicted classes.
        """
        y_proba_aux = proba_matrix[idx]  # (n_samples, n_classes)

        y_proba_total = self.state_ + y_proba_aux  # Add the new idx proba to the state proba

        # Predict the class with the max probability
        y_pred = self.classes_.take((np.argmax(y_proba_total, axis=1)), axis=0)  # (n_samples)

        return y_pred

    def pred_max_voted_state(self):
        """Predict the class with more votes, with the current sub-ensemble.

        Returns
        -------
        y_pred : array of shape = [n_samples]
            The predicted classes.
        """
        y_pred = np.empty(self.n_samples_)  # (n_samples)
        for i in range(self.n_samples_):
            y_pred[i] = self.state_[i].argmax()  # Take the max voted class

        return y_pred

    def pred_max_proba_state(self):
        """Predict the class with more probability, with the current sub-ensemble.

        Returns
        -------
        y_pred : array of shape = [n_samples]
            The predicted classes.
        """
        y_pred = self.classes_.take((np.argmax(self.state_, axis=1)), axis=0)  # (n_samples)

        return y_pred


class PruningStateMaxVoted(PruningState):
    """A PruningState based on max voted criteria.

    Attributes
    ----------
    pred_matrix_ : array of shape = [n_estimators, n_samples]
       The prediction matrix for the input samples.
    """
    def __init__(self):
        super(PruningStateMaxVoted, self).__init__()

        self.pred_matrix_ = None

    def start(self, ensemble_pruning, X, y):
        super(PruningStateMaxVoted, self).start(ensemble_pruning, X, y)
        self.state_ = np.zeros((self.n_samples_, self.n_classes_), dtype=int)

        self.pred_matrix_ = ensemble_pruning.build_pred_matrix(X)
        return

    def update(self, idx, score):
        super(PruningStateMaxVoted, self).update(idx, score)

        y_aux = self.pred_matrix_[idx]  # (n_samples)
        for i, pred in enumerate(y_aux):
            self.state_[i, pred] += 1  # add a vote for the selected class

        return

    def partial_score(self, idx):
        y_pred = self.pred_max_voted(idx, self.pred_matrix_)
        score = accuracy_score(self.y_, y_pred)
        return score


class PruningStateProba(PruningState):
    """Base class for all probability based PruningStates.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Attributes
    ----------
    prob_matrix_ : array of shape = [n_estimators, n_samples, n_classes]
       The probability matrix for the input samples.
    """
    def __init__(self):
        super(PruningStateProba, self).__init__()

        self.prob_matrix_ = None

    @abstractmethod
    def start(self, ensemble_pruning, X, y):
        super(PruningStateProba, self).start(ensemble_pruning, X, y)
        self.state_ = np.zeros((self.n_samples_, self.n_classes_))

        self.prob_matrix_ = ensemble_pruning.build_prob_matrix(X)
        return

    @abstractmethod
    def update(self, idx, score):
        super(PruningStateProba, self).update(idx, score)
        y_aux = self.prob_matrix_[idx]  # (n_samples, n_classes)
        self.state_ += y_aux  # Update the probability
        return


class PruningStateMaxProba(PruningStateProba):
    """A PruningState based on max probability criteria."""
    def __init__(self):
        super(PruningStateMaxProba, self).__init__()

    def start(self, ensemble_pruning, X, y):
        super(PruningStateMaxProba, self).start(ensemble_pruning, X, y)
        return

    def update(self, idx, score):
        super(PruningStateMaxProba, self).update(idx, score)
        return

    def partial_score(self, idx):
        y_pred = self.pred_max_proba(idx, self.prob_matrix_)
        score = accuracy_score(self.y_, y_pred)
        return score


class PruningStateComplementary(PruningStateProba):
    """A PruningState that prioritize the estimators that better
    classifies the wrong classified samples for the ensemble.

    Attributes
    ----------
    hits_ : array of shape = [n_samples]
       The array that stores the hit or miss predictions for the current sub-ensemble.

    hit_matrix_ : array of shape = [n_estimators, n_samples]
       The matrix to store the hit or miss predictions for all the estimators.
    """
    def __init__(self):
        super(PruningStateComplementary, self).__init__()

        self.hits_ = None
        self.hit_matrix_ = None

    def start(self, ensemble_pruning, X, y):
        super(PruningStateComplementary, self).start(ensemble_pruning, X, y)

        self.hits_ = np.zeros(self.n_samples_)  # Hits per sample
        self.hit_matrix_ = self._build_hit_matrix()  # Hit Matrix (n_estimators, n_samples)

        return

    def update(self, idx, score):
        super(PruningStateComplementary, self).update(idx, score)

        y_pred = self.pred_max_proba_state()
        self.scores_[-1] = accuracy_score(self.y_, y_pred)  # Update with actual score

        # Update hit array
        for i in range(self.n_samples_):
            if y_pred[i] == self.y_[i]:
                self.hits_[i] = 1
            else:
                self.hits_[i] = 0

        return

    def partial_score(self, idx):
        idx_hits = self.hit_matrix_[idx]
        s = 0
        c = 0

        for i in range(self.n_samples_):
            if self.hits_[i] == 0:  # Check if wrong classified for the ensemble
                c += 1
                if idx_hits[i] == 1:  # Check if correct classified for the classifier
                    s += 1

        if c == 0:  # If there are no wrong classified samples
            score = np.count_nonzero(idx_hits)
        else:
            score = s / c  # Compute score

        return score

    def _build_hit_matrix(self):
        """Build and return the hit matrix"""
        n_estimators = self.prob_matrix_.shape[0]  # Get n_estims
        n_samples = self.prob_matrix_.shape[1]  # Get n_samples

        # Hit Matrix (n_estimators, n_samples)
        hit_matrix = np.zeros((n_estimators, n_samples))

        for i in range(n_estimators):  # For each estimator
            y_pred = self.classes_.take((np.argmax(self.prob_matrix_[i], axis=1)), axis=0)
            for j in range(n_samples):  # Check predictions
                if y_pred[j] == self.y_[j]:
                    hit_matrix[i][j] = 1  # Write 1 for each hit

        return hit_matrix


class PruningStateUWA(PruningStateComplementary):
    """A PruningState that use the Uncertainty Weighted Accuracy (UWA) metric.

    Attributes
    ----------
    idx_ensemble_ : list
       The current sub-ensemble estimators indexes.

    NT_ : array of shape = [n_samples]
       The fractions of classifiers in the current sub-ensemble that classify an instance correctly.

    NF_ : array of shape = [n_samples]
       The fractions of classifiers in the current sub-ensemble that classify an instance incorrectly.
    """
    def __init__(self):
        super(PruningStateProba, self).__init__()

        self.idx_ensemble_ = None
        self.NT_ = None
        self.NF_ = None

    def start(self, ensemble_pruning, X, y):
        super(PruningStateUWA, self).start(ensemble_pruning, X, y)

        self.idx_ensemble_ = []

        self.NT_ = np.array([1.] * self.n_samples_)
        self.NF_ = np.array([1.] * self.n_samples_)

        return

    def update(self, idx, score):
        super(PruningStateUWA, self).update(idx, score)
        self.idx_ensemble_.append(idx)  # add the new idx to the sub-ensemble

        for i in range(self.n_samples_):
            self.NT_[i] = self._NT(i)
            self.NF_[i] = 1 - self.NT_[i]

        return

    def partial_score(self, idx):
        score = 0
        idx_hits = self.hit_matrix_[idx]  # (n_samples)

        for i in range(self.n_samples_):
            if self.hits_[i] == 0:
                if idx_hits[i] == 0:
                    score -= self.NT_[i]  # ff
                else:
                    score += self.NT_[i]  # tf
            else:
                if idx_hits[i] == 0:
                    score -= self.NF_[i]  # ft
                else:
                    score += self.NF_[i]  # tt

        return score

    def _NT(self, sample_idx):
        """Return the fraction of classifiers in the current sub-ensemble that
        classify the given sample correctly"""
        all_hits = self.hit_matrix_[:, sample_idx]  # Take hit array for the current sample
        hits = np.take(all_hits, self.idx_ensemble_)  # Take only the hits for the actual ensemble
        nt = np.count_nonzero(hits) / len(self.idx_ensemble_)  # self.n_idx_
        return nt
