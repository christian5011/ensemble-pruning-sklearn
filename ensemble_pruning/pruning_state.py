"""Pruning state implementations"""

# Author: Christian Messina <christian.messina.val@gmail.com>
# License:

from __future__ import division

import numpy as np
from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score


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