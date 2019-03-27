import numpy as np
from .base import CompetitiveNetwork
from .utils import default_learning_rate_decay_function, default_sigma_decay_function

class SOM(CompetitiveNetwork):
  """Self organizing map"""
  def __init__(self, n_rows, n_cols):
    super().__init__(n_rows, n_cols)

  def find_nearest_node(self, x):
    return np.argmin(np.sum((self._competitive_layer_weights - x) ** 2, axis = 1) + self._bias ** 2)

  def _neighborhood_functions(self, win_idx, sigma):
    win_x, win_y = win_idx // self._n_cols, win_idx % self._n_cols
    X1, X2 = np.meshgrid(np.arange(self._n_cols), np.arange(self._n_rows))
    distance = np.reshape(np.sqrt((X1 - win_y) ** 2 + (X2 - win_x) ** 2), (self._n_nodes,))
    neighbors = np.zeros(self._n_nodes)
    if self._neighborhood == "bubble":
      neighbors[distance <= sigma] = 1
    elif self._neighborhood == "gaussian":
      neighbors = np.exp(- distance / (2 * (sigma ** 2)))
    return neighbors

  def update(self, x, batch):
    """Updating competitive layer weights bases on winning node with sample x
    """
    learning_rate = self._learnining_rate_decay_funtion(self._initial_learnining_rate, self._learnining_decay_rate, batch)
    sigma = self._sigma_decay_function(self._initial_sigma, self._sigma_decay_rate, batch)
    winning_node_idx = self.find_nearest_node(x)
    neighbors = self._neighborhood_functions(winning_node_idx, sigma)
    
    # Neighbor need to have shape of (n_nodes, 1)
    neighbors = neighbors.reshape(-1 ,1)
    
    # Updating competitive layer weights
    self._competitive_layer_weights = self._competitive_layer_weights + (x - self._competitive_layer_weights) * neighbors * learning_rate

  def fit(self, X, num_iters, batch_size = 32, neighborhood = "bubble",
          learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          bias = False, verbose = 0):
    """Fit the model according to the input data
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.

    num_iters : int
      Number of iterations.

    batch_size : int
      Number of samples.

    Returns
    -------
    self : object
      Return self.
    """
    if len(X.shape) != 2:
      raise Exception("Dataset need to be 2 dimensions")
    else:
      n_samples, n_features = X.shape
      
    self._competitive_layer_weights = np.random.rand(self._n_nodes, n_features)
    self._initial_learnining_rate = learning_rate
    self._initial_sigma = sigma
    self._learnining_decay_rate = learnining_decay_rate
    self._sigma_decay_rate = sigma_decay_rate
    self._bias = np.zeros(self._n_nodes)
    self._neighborhood = neighborhood
    if learning_rate_decay_function:
      self._learnining_rate_decay_funtion = learning_rate_decay_function
    else:
      self._learnining_rate_decay_funtion = default_learning_rate_decay_function

    if sigma_decay_function:
      self._sigma_decay_function = sigma_decay_function
    else:
      self._sigma_decay_function = default_sigma_decay_function
    
    for i in range(num_iters):
      sample_idx = np.random.randint(0, n_samples)
      x = X[sample_idx]
      n_batchs = i // batch_size
      self.update(x = x, batch = n_batchs)

  def predict(self, X):
    pass