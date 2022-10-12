"""This is the example module.

This module does stuff.
"""
__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Jalil Nourisa'

import os
import time
import sys
import numpy as np
import itertools
import pathlib
from pathos.pools import ParallelPool as Pool

from sklearn import tree
from sklearn import ensemble
from sklearn import base
from sklearn import utils
from sklearn import inspection
from sklearn import metrics

dir_main = os.path.join(pathlib.Path(__file__).parent.resolve(),'..')
sys.path.insert(0, dir_main)

from geneRNI import tools


#TODO: lag h: can be more than 1. can be in the format of hour/day 
#TODO: how to make the static data comparable to dynamic data
#TODO: add alpha to data processing and fit and score
#TODO: scripts to manage continuous integration (testing on Linux and Windows)


# TODOC: pathos is used instead of multiprocessing, which is an external dependency. 
#        This is because multiprocessing uses pickle that has problem with lambda function.

def network_inference(Xs, ys, gene_names, param, param_unique = None, Xs_test=None, ys_test=None, verbose=True):
    """ Determines links of network inference
    If the ests are given, use them instead of creating new ones.

    Xs -- 
    """
    n_genes = len(ys)
    
    if param_unique == None:
        ests = [GeneEstimator(**param) for i in range(n_genes)]
    elif isinstance(param_unique,dict):
        ests = [GeneEstimator(**{**param,**param_unique}) for i in range(n_genes)]  
    else: 
        ests = [GeneEstimator(**{**param,**param_unique[i]}) for i in range(n_genes)]  
    fits = [ests[i].fit(X,y) for i, (X, y) in enumerate(zip(Xs,ys))]
    
    # train score
    train_scores = [ests[i].score(X,y) for i, (X, y) in enumerate(zip(Xs,ys))]
    tools.verboseprint(verbose, f'\nnetwork inference: train score, mean: {np.mean(train_scores)} std: {np.std(train_scores)}')
    # print(f'\nnetwork inference: train score, mean: {np.mean(train_scores)} std: {np.std(train_scores)}')
    # oob score
    if param['estimator_t'] == 'RF':
        oob_scores = [est.est.oob_score_ for est in ests]  
        tools.verboseprint(verbose,f'network inference: oob score (only RF), mean: {np.mean(oob_scores)} std: {np.std(oob_scores)}')      
    else:
        oob_scores = None
    # test score
    if Xs_test is not None or ys_test is not None:
        test_scores = [ests[i].score(X,y) for i, (X, y) in enumerate(zip(Xs_test,ys_test))]
        tools.verboseprint(verbose,f'network inference: test score, mean: {np.mean(test_scores)} std: {np.std(test_scores)}')
    else:
        test_scores = None
    # feature importance
    # if Xs_test is not None:
    #     print('Permutation based feature importance')
    #     links_p = [ests[i].compute_feature_importance_permutation(X,y) for i, (X, y) in enumerate(zip(Xs_test,ys_test))]
    # else:
    #     print('Variance based feature importance')
    #     links_v = [ests[i].compute_feature_importances_tree() for i,_ in enumerate(ys)]
    # links = links_p
    links = [ests[i].compute_feature_importances_tree() for i,_ in enumerate(ys)]
    links_df = tools.Links.format(links, gene_names)
    # links = None
    return ests, train_scores, links_df, oob_scores, test_scores
class GenesEstimator(base.BaseEstimator,base.RegressorMixin):
    """The docstring for a class should summarize its behavior and list the public methods and instance variables """
    def __init__(self, estimator=None, alphas = 0, **params):
        '''args should all be keyword arguments with a default value -> kwargs should be all the keyword params of all regressors with values'''
        '''they should not be documented under the “Attributes” section, but rather under the “Parameters” section for that estimator.'''
        '''every keyword argument accepted by __init__ should correspond to an attribute on the instance'''
        '''There should be no logic, not even input validation, and the parameters should not be changed. The corresponding logic should be put where the parameters are used, typically in fit'''
        '''algorithm-specific unit tests,'''
        # self.alpha = alpha
        self.params = params
        self.estimator = estimator
        self.alphas = alphas
        self.ests = []
        # self._required_parameters = () #estimators also need to declare any non-optional parameters to __init__ in the
    def apply_alpha(self, ys):

        if self.alphas == 0:
            ys = [[ys[i,j](0) for j in range(ys.shape[1])] for i in range(ys.shape[0])]
        else:
            ys = [[ys[i,j](self.alphas[j]) for j in range(ys.shape[1])] for i in range(ys.shape[0])]
        
        return np.array(ys)
    def check_X_y(self, X, ys):
        utils.check_array(X)
        utils.indexable(X)
        for i in range(ys.shape[1]):
            y = ys[:,i]
            utils.check_X_y(X,y)
            utils.indexable(y)
    def fit(self, X, ys):
        """ fit X to y
        X -- Array-like of shape (n_samples, n_features)
        ys, where y -- Array-like of shape (n_samples,)
        """
        '''Attributes that have been estimated from the data must always have a name ending with trailing underscore'''
        '''The estimated attributes are expected to be overridden when you call fit a second time.'''
        
        # apply alpha to y
        ys = self.apply_alpha(ys)
        self.check_X_y(X,ys)

        for i in range(ys.shape[1]): #TODO: this might need to go to set_params
            # if self.params is not None:
            #     if utils.indexable(self.params): # seperate parameter sets are given for genes
            #         params = self.params[i]
            #     else:
            #         params = self.params
            # else:
            #     params = {}
            params = {}
            if self.estimator is None:
                est = ensemble.RandomForestRegressor(oob_score = True,**params)
            else:
                est = self.estimator(**params)
            self.ests.append(est)

        for i in range(ys.shape[1]):
            y = ys[:,i]
            self.ests[i].fit(X,y)
        self.X_ = X
        self.y_ = ys

        return self
    def check_is_fitted(self):
        for est in self.ests:
            utils.validation.check_is_fitted(est)
    def predict(self,Xs):
        """ """ 
        utils.validation.check_is_fitted(self.ests)

        return [est.predict(X) for est,X in zip(self.ests,Xs)]
    def score(self, X, ys): 
        """ """
        # apply alpha to y
        ys = self.apply_alpha(ys)
        self.check_X_y(X,ys)
        self.check_is_fitted()

        return [self.ests[i].score(X,ys[:,i]) for i in range(ys.shape[1])]

    # def score_cv(self,X,ys, cv=4):
    #     """cross validation score. """
    #     est.set_params(**params)
    #     if use_oob_flag:
    #         fit = est.fit(X,y)
    #         score = est.est.oob_score_
    #     else:
    #         cv_a = model_selection.ShuffleSplit(n_splits=cv, test_size=(1/cv)) 
    #         scores = model_selection.cross_val_score(est, X, y, cv=cv_a)
    #         score = np.mean(scores)
            
    def compute_feature_importances_tree(self): 
        """Computes variable importances from a trained tree-based model. Deprecated"""
        
        if isinstance(self.est, tree.BaseDecisionTree):
            return self.est.tree_.compute_feature_importances(normalize=False)
        else:
            importances = [e.tree_.compute_feature_importances(normalize=False)
                           for e in self.est.estimators_]
            importances = np.array(importances)
            return np.sum(importances,axis=0) / len(importances)
    def permutation_importance(self, X_test=None, y_test=None ,n_repeats=20):
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
    def compute_feature_importance_permutation (self, X_test, y_test):
        """ Determines importance of regulatory genes """
        vi,_ = self.permutation_importance(X_test, y_test)
        # vi = self.compute_feature_importances_tree()
        # Normalize importance scores
        #TODO: double check if the normalization is valid
        vi_sum = sum(vi)
        if vi_sum > 0:
            vi = vi / vi_sum
        return vi
    def get_params(self,deep=True):
        """ The get_params function takes no arguments and returns a dict of 
        the __init__ parameters of the estimator, together with their values. 

        """
        return {'estimator': self.estimator, 'alphas': self.alphas, **self.params}
    def set_params(self, **parameters):
        """ """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def _more_tags(self):
        """ """
        # if self.estimator_t == 'HGB':
        #     allow_nan = True 
        # else:
        #     allow_nan = False
        return {'requires_fit': True, 'allow_nan': allow_nan, 'multioutput': True, 
            'requires_y': True,}

    
   