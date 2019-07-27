import numpy as np

class CompetitiveNetwork(object):
  """Base class for competitive networks"""
  def __init__(self, n_rows, n_cols, random_state):
    self._n_rows = n_rows
    self._n_cols = n_cols
    self._n_nodes = n_rows * n_cols
    self._random_state = random_state

  def _get_random_state(self):
    max_seed = np.iinfo(np.int32).max
    np.random.seed(self._random_state)
    self._random_state = np.random.randint(max_seed)

    return self._random_state

  def fit(self, X, y):
    """Fit model."""
    pass

  def predict(self, X):
    """Predict using fitted model"""
    pass