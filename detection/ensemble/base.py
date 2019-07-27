import numpy as np

class CombinationNetworksBase(object):
  """Base class for combination of networks classes"""
  def __init__(self, n_estimators, size_of_estimator, random_state):
    self._n_estimators = n_estimators
    self._size_of_estimator = size_of_estimator
    self._models = []
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