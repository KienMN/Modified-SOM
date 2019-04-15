import numpy as np
from .base import CompetitiveNetwork
from .utils import default_learning_rate_decay_function
from .utils import default_sigma_decay_function
from .utils import weights_initialize

class SOM(CompetitiveNetwork):
  """Self organizing map"""
  def __init__(self, n_rows, n_cols):
    super().__init__(n_rows, n_cols)
    self._quantization_error = np.array([])

  def find_nearest_node(self, x):
    return np.argmin(np.sum((self._competitive_layer_weights - x) ** 2, axis = 1) + self._bias ** 2)

  def neighborhood_functions(self, win_idx, sigma):
    win_x, win_y = win_idx // self._n_cols, win_idx % self._n_cols
    X1, X2 = np.meshgrid(np.arange(self._n_cols), np.arange(self._n_rows))
    distance = np.reshape(np.sqrt((X1 - win_y) ** 2 + (X2 - win_x) ** 2), (self._n_nodes,))
    neighbors = np.zeros(self._n_nodes)
    if self._neighborhood == "bubble":
      neighbors[distance <= sigma] = 1
    elif self._neighborhood == "gaussian":
      neighbors = np.exp(- distance / (2 * (sigma ** 2)))
    return neighbors

  def unsup_fitting(self, X, num_iters, batch_size):
    n_samples = X.shape[0]
    for i in range(num_iters):
      sample_idx = np.random.randint(0, n_samples)
      x = X[sample_idx]
      n_batchs = i // batch_size
      self.update(x = x, batch = n_batchs)
      self._quantization_error = np.append(self._quantization_error, self.quantization_error(X))

  def update(self, x, batch):
    """Updating competitive layer weights bases on winning node with sample x
    """
    learning_rate = self._learnining_rate_decay_funtion(self._initial_learnining_rate, self._learnining_decay_rate, batch)
    sigma = self._sigma_decay_function(self._initial_sigma, self._sigma_decay_rate, batch)
    winning_node_idx = self.find_nearest_node(x)
    neighbors = self.neighborhood_functions(winning_node_idx, sigma)
    
    # Neighbor need to have shape of (n_nodes, 1)
    neighbors = neighbors.reshape(-1 ,1)
    
    # Updating competitive layer weights
    self._competitive_layer_weights = self._competitive_layer_weights + ((x - self._competitive_layer_weights) * learning_rate) * neighbors

    # Updating bias
    if self._conscience:
      self._bias[:winning_node_idx] *= 0.9
      self._bias[winning_node_idx] += 0.1
      self._bias[winning_node_idx + 1:] *= 0.9
    
  def fit(self, X, num_iters, batch_size = 32, neighborhood = "bubble",
          learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = 0):
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
      n_features = X.shape[1]
      
    self._competitive_layer_weights = np.random.rand(self._n_nodes, n_features)
    self._initial_learnining_rate = learning_rate
    self._initial_sigma = sigma
    self._learnining_decay_rate = learnining_decay_rate
    self._sigma_decay_rate = sigma_decay_rate
    self._conscience = conscience
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

    self.unsup_fitting(X, num_iters, batch_size)
    return self

  def predict(self, X):
    pass

  def quantization_error(self, X):
    """
    Returns quantization error computed from the average distance of input vectors, x to its best matching unit
    """
    if len(X.shape) != 2:
      raise Exception("Dataset need to be 2 dimensions")
    error = 0
    for x in X:
      error += np.sqrt(np.sum((x - self._competitive_layer_weights[self.find_nearest_node(x)]) ** 2))
    return error / len(X)

class CombineSomLvq(SOM):
  """Combination of SOM and LVQ"""
  def __init__(self, n_rows, n_cols):
    super().__init__(n_rows, n_cols)

  def label_nodes(self, X, y, labels_init):
    self._nodes_label = np.zeros(self._n_nodes).astype(np.int8)
    m = 20
    for i in range (self._n_nodes):
      near_samples_idx = np.argpartition(np.sum((self._competitive_layer_weights[i] - X) ** 2, axis = 1), m)[:m]
      l = np.argmax(np.bincount(y[near_samples_idx]))
      self._nodes_label[i] = l

  def sup_fitting(self, X, y, num_iters, batch_size, n_trained_batchs):
    n_samples = X.shape[0]
    for i in range(num_iters):
      sample_idx = np.random.randint(0, n_samples)
      x = X[sample_idx]
      y_i = y[sample_idx]
      n_batchs = n_trained_batchs + i // batch_size
      self.sup_update(x = x, y = y_i, batch = n_batchs)
      self._quantization_error = np.append(self._quantization_error, self.quantization_error(X))

  def sup_update(self, x, y, batch):
    """Updating competitive layer weights bases on winning node with sample x
    """
    learning_rate = self._learnining_rate_decay_funtion(self._initial_learnining_rate, self._learnining_decay_rate, batch)
    sigma = self._sigma_decay_function(self._initial_sigma, self._sigma_decay_rate, batch)
    winning_node_idx = self.find_nearest_node(x)
    neighbors = self.neighborhood_functions(winning_node_idx, sigma)
    
    # Neighbor need to have shape of (n_nodes, 1)
    neighbors = neighbors.reshape(-1 ,1)
    
    # Sign
    sign = np.ones((self._n_nodes, 1))
    sign[self._nodes_label == y] = 1
    sign[self._nodes_label == y] = -1 / 3

    # Updating competitive layer weights
    self._competitive_layer_weights = self._competitive_layer_weights + ((x - self._competitive_layer_weights) * learning_rate) * neighbors * sign

    # Updating bias
    if self._conscience:
      self._bias[:winning_node_idx] *= 0.9
      self._bias[winning_node_idx] += 0.1
      self._bias[winning_node_idx + 1:] *= 0.9

  def fit(self, X, y, weights_init = None, labels_init = None,
          unsup_num_iters = 100, unsup_batch_size = 32,
          sup_num_iters = 100, sup_batch_size = 32,
          neighborhood = "bubble",
          learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = 0):
    
    if len(X.shape) != 2:
      raise Exception("Dataset need to be 2 dimensions")
      
    if weights_init == 'sample' or weights_init == 'pca':
      self._competitive_layer_weights = weights_initialize(X, self._n_rows, self._n_cols, method = weights_init)
    else:
      print('No weights init specified, using random instead')
      self._competitive_layer_weights = weights_initialize(X, self._n_rows, self._n_cols, method = 'random')

    self._initial_learnining_rate = learning_rate
    self._initial_sigma = sigma
    self._learnining_decay_rate = learnining_decay_rate
    self._sigma_decay_rate = sigma_decay_rate
    self._conscience = conscience
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
    
    # Unsupervised learning phase
    self.unsup_fitting(X, unsup_num_iters, unsup_batch_size)

    # Supervised learning phase
    self.label_nodes(X, y, labels_init)
    self.sup_fitting(X, y, sup_num_iters, sup_batch_size, n_trained_batchs = unsup_num_iters // unsup_batch_size)
    self.label_nodes(X, y, labels_init)

  def predict(self, X):
    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples).astype(np.int8)
    for i in range (n_samples):
      winning_node_idx = self.find_nearest_node(X[i])
      y_pred[i] = self._nodes_label[winning_node_idx]
    return y_pred