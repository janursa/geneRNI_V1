
"""This is the example module.

This module does stuff.
"""

__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

import os
import pathlib
import sys
from typing import Optional, Any

import numpy as np
from sklearn import base
from sklearn import inspection
from sklearn import utils

from geneRNI import tools
from geneRNI.data import Data
from geneRNI.models import get_estimator_wrapper
from geneRNI.utils import is_lambda_function

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(), '..')
sys.path.insert(0, dir_main)  # TODO: not recommended (let's make a setup.py file instead)


#TODO: lag h: can be more than 1. can be in the format of hour/day 
#TODO: how to make the static data comparable to dynamic data
#TODO: add alpha to data processing and fit and score
#TODO: scripts to manage continuous integration (testing on Linux and Windows)


# TODOC: pathos is used instead of multiprocessing, which is an external dependency. 
#        This is because multiprocessing uses pickle that has problem with lambda function.

def network_inference(data: Data, gene_names, param, param_unique=None, verbose=True, output_dir=None):
    """ Determines links of network inference
    If the ests are given, use them instead of creating new ones.
    """
    n_genes = len(data)
    
    if param_unique is None:
        ests = [GeneEstimator(**param) for _ in range(n_genes)]
    elif isinstance(param_unique, dict):
        ests = [GeneEstimator(**{**param, **param_unique}) for _ in range(n_genes)]
    else: 
        ests = [GeneEstimator(**{**param, **param_unique[i]}) for i in range(n_genes)]

    links = []
    train_scores, test_scores, oob_scores = [], [], []
    for i in range(n_genes):
        X_train, X_test, y_train, y_test = data[i]

        # Estimate train and test scores to assess generalization abilities
        ests[i].fit(X_train, y_train)
        train_scores.append(ests[i].score(X_train, y_train))
        test_scores.append(ests[i].score(X_test, y_test))
        if param['estimator_t'] == 'RF':
            oob_scores.append(ests[i].est.oob_score_)

        # Actual network inference, using all the available data
        ests[i].fit(np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0))
        links.append(ests[i].compute_feature_importances())

    # Show scores
    tools.verboseprint(verbose, f'\nnetwork inference: train score, mean: {np.mean(train_scores)} std: {np.std(train_scores)}')
    tools.verboseprint(
        verbose,
        f'network inference: test score, mean: {np.mean(test_scores)} std: {np.std(test_scores)}')
    if len(oob_scores) > 0:
        tools.verboseprint(
            verbose,
            f'network inference: oob score (only RF), mean: {np.mean(oob_scores)} std: {np.std(oob_scores)}')

    # Save predicted regulatory relations
    links_df = tools.Links.format(links, gene_names)
    if output_dir is not None:
        links_df.to_csv(os.path.join(output_dir, 'links.txt'))

    return ests, train_scores, links_df, oob_scores, test_scores


class GeneEstimator(base.RegressorMixin):
    """The docstring for a class should summarize its behavior and list the public methods and instance variables """

    def __init__(self, estimator_t: str, alpha: float = 0., **params):
        '''args should all be keyword arguments with a default value -> kwargs should be all the keyword params of all regressors with values'''
        '''they should not be documented under the “Attributes” section, but rather under the “Parameters” section for that estimator.'''
        '''every keyword argument accepted by __init__ should correspond to an attribute on the instance'''
        '''There should be no logic, not even input validation, and the parameters should not be changed. The corresponding logic should be put where the parameters are used, typically in fit'''
        '''algorithm-specific unit tests,'''
        # self.alpha = alpha
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.params: dict = params
        self.estimator_t: str = estimator_t
        self.alpha: float = alpha
        self.est = None

        # estimators also need to declare any non-optional parameters to __init__ in the
        # self._required_parameters = ('allow_nan')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GeneEstimator':
        """ fit X to y
        X -- Array-like of shape (n_samples, n_features)
        y -- Array-like of shape (n_samples,)
        kwargs -- Optional data-dependent parameters
        """
        '''Attributes that have been estimated from the data must always have a name ending with trailing underscore'''
        '''The estimated attributes are expected to be overridden when you call fit a second time.'''
        
        # apply alpha to y
        try:  # TODO: bad design
            y = [y_i(self.alpha) for y_i in y]
        except:
            pass

        utils.check_array(X)
        utils.check_X_y(X, y)
        utils.indexable(X)
        utils.indexable(y)

        # Check this. https://scikit-learn.org/stable/developers/utilities.html#developers-utils
        utils.assert_all_finite(X)
        utils.assert_all_finite(y)

        self.X_ = X
        self.y_ = y
        self.est = get_estimator_wrapper(self.estimator_t).new_estimator()
        self.est.fit(X, self.check_target(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ """
        utils.validation.check_is_fitted(self.est)
        return self.est.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """ """
        # apply alpha to y
        try:
            y = [y_i(self.alpha) for y_i in y]
        except:
            pass
        utils.validation.check_is_fitted(self.est)
        return self.est.score(X, self.check_target(y))

    def compute_feature_importances(self) -> np.ndarray:
        """Computes variable importances from a trained model."""
        return get_estimator_wrapper(self.estimator_t).compute_feature_importances(self.est)

    def permutation_importance(
            self,
            X_test: Optional[np.ndarray] = None,
            y_test: Optional[np.ndarray] = None,
            n_repeats: int = 20
    ):
        """Computes variable importances for a trained model
        In case X and y are not given, the process is done on the train data.
        
        n_repeats -- number of times a feature is randomly shuffled 
        """
        """When two features are correlated and one of the features is permuted, the model will still have access to the 
        feature through its correlated feature. This will result in a lower importance value for both features, where 
        they might actually be important. One way to handle this is to cluster features that are correlated and only 
        keep one feature from each cluster. This strategy is explored in the following example: Permutation Importance
         with Multicollinear or Correlated Features."""

        utils.validation.check_is_fitted(self.est)

        if X_test is None or y_test is None:
            print("Permutation importance on the train samples")
            r = inspection.permutation_importance(self.est, self.X_, self.y_, n_repeats=n_repeats)
        else:
            print("Permutation importance on the test samples")
            y_test = [y_i(self.alpha) for y_i in y_test]
            r = inspection.permutation_importance(self.est, X_test, y_test, n_repeats=n_repeats)
        return r['importances_mean'], r['importances_std']

    def compute_feature_importance_permutation(self, X_test: np.ndarray, y_test: np.ndarray):
        """ Determines importance of regulatory genes """
        vi, _ = self.permutation_importance(X_test, y_test)
        # vi = self.compute_feature_importances_tree()
        # Normalize importance scores
        #TODO: double check if the normalization is valid
        vi_sum = sum(vi)
        if vi_sum > 0:
            vi = vi / vi_sum
        return vi

    def check_target(self, y: Any) -> np.ndarray:
        new_y = np.empty(len(y), dtype=float)
        for i in range(len(y)):
            if is_lambda_function(y[i]):
                new_y[i] = y[i](self.alpha)  # TODO: is this correct?
            else:
                new_y[i] = y[i]
        return new_y

    def get_params(self, deep: bool = True) -> dict:
        """ The get_params function takes no arguments and returns a dict of 
        the __init__ parameters of the estimator, together with their values. 

        """
        return {'estimator_t': self.estimator_t, 'alpha': self.alpha, **self.params}

    def set_params(self, **parameters) -> 'GeneEstimator':
        """ """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _validate_data(self, X, y):
        pass

    def _more_tags(self) -> dict:
        """ """
        if self.estimator_t == 'HGB':
            allow_nan = True 
        else:
            allow_nan = False
  
        return {
            'requires_fit': True,
            'allow_nan': allow_nan,
            'multioutput': True,
            'requires_y': True
        }

