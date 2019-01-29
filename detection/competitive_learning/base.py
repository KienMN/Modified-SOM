import numpy as np

class CompetitiveNetwork(object):
  """Base class for competitive networks"""
  def __init__(self, n_rows, n_cols):
    self._n_rows = n_rows
    self._n_cols = n_cols

  def fit(self, X, y):
    """Fit model."""
    pass

  def predict(self):
    """Predict using fitted model"""
    pass