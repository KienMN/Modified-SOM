import numpy as np

class CombinationNetworksBase(object):
  """Base class for combination of networks classes"""
  def __init__(self, n_estimators, size_of_estimator):
    self._n_estimators = n_estimators
    self._size_of_estimator = size_of_estimator
    self._models = []

  def fit(self, X, y):
    """Fit model."""
    pass

  def predict(self, X):
    """Predict using fitted model"""
    pass