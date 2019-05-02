import numpy as np
from .base import CombinationNetworksBase
from .utils import weighted_sampling
from ..competitive_learning.som import CombineSomLvq

class CombinationSomLvqNetworks(CombinationNetworksBase):
  """Combination of SOM-LVQ networks.
  
  Parameters
  ----------
  n_estimators : int
    Number of SOM-LVQ networks in combination.

  size_of_estimator : int
    Size of each SOM-LVQ networks.

  Attributes
  ----------
  _models : list
    List of models in combination.
  """
  
  def __init__(self, n_estimators, size_of_estimator):
    super().__init__(n_estimators, size_of_estimator)
    for _ in range (n_estimators):
      self._models.append(CombineSomLvq(n_rows = size_of_estimator, n_cols = size_of_estimator))

  def fit(self, X, y, subset_size = 0.25, weights_init = None, labels_init = None,
          unsup_num_iters = 100, unsup_batch_size = 32,
          sup_num_iters = 100, sup_batch_size = 32,
          neighborhood = "bubble",
          learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = 0):
    """Fit the models according to the input data.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.

    y : 1D numpy array, shape (n_samples,)
      Training label vector, where n_samples in the number of samples.

    subset_size : float, default: 0.25
      Size of subset according to size of the dataset.

    weights_init : str, options: ['random', 'sample', 'pca'], default: random
      Strategies to initialize the weights of the neurons.

    labels_init : str, default: None
      Criteria to initialize the labels of the neurons.

    unsup_num_iters : int
      Number of iterations that unsupervised learning phase passes.

    unsup_batch_size : int
      Number of samples in the same batch during unsupervised learning phase,
      that uses the same learning rate and neighborhood's radius.

    sup_num_iters : int
      Number of iterations that supervised learning phase passes.

    sup_batch_size : int
      Number of samples in the same batch during supervised learning phase,
      that uses the same learning rate and neighborhood's radius.

    neighborhood : str, options: ['bubble', 'gaussian']
      Neighborhood function that is used to compute neighborhood coefficients.

    learning_rate : float, default: 0.5
      The learning rate that is used during training process.

    learning_decay_rate : float, default: 1
      The rate that is used to decrease the learning rate after each batches.

    learning_rate_decay_function : function, default: None
      The function that is used to decrease the learning rate after each batches.

    sigma : float, default: 1
      The radius that is used to compute neighborhood coefficients.

    sigma_decay_rate : float, default: 1
      The rate that is used to decrease the sigma after each batches.

    sigma_decay_function : function, default: None
      The function that is used to decrease the sigma after each batches.

    conscience : boolean, default: False
      The technique that tracks the frequency of winning of each neurons to avoid a neuron from winning too many times.

    verbose : boolean, optional, default False
      Verbose mode when fitting the models.

    Returns
    -------
    self : object
      Return self.
    """
    n_samples = X.shape[0]
    for i, model in enumerate(self._models):
      subset_idx = np.random.randint(0, n_samples, int(subset_size * n_samples))
      X_subset = X[subset_idx]
      y_subset = y[subset_idx]
      if verbose:
        print('Model {}/{}:'.format(i + 1, self._n_estimators))
      model.fit(X_subset, y_subset, weights_init = weights_init, labels_init = labels_init,
                unsup_num_iters = unsup_num_iters, unsup_batch_size = unsup_batch_size,
                sup_num_iters = sup_batch_size, sup_batch_size = sup_batch_size,
                neighborhood = neighborhood,
                learning_rate = learning_rate, learnining_decay_rate = learnining_decay_rate,
                learning_rate_decay_function = learning_rate_decay_function,
                sigma = sigma, sigma_decay_rate = sigma_decay_rate, sigma_decay_function = sigma_decay_function,
                conscience = conscience, verbose = verbose)
    return self

  def predict(self, X, crit = 'max-voting-uniform'):
    """Predict using the fitted model.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Input vectors, where n_samples is the number of samples and n_features is the number of features.

    crit : str, option: ['max-voting-uniform', 'max-voting-weight', 'max-mean-confidence-score']
      Criteria to combine results of models.

    Returns
    -------
    y_pred : 1D numpy array, shape (n_samples,)
      Predicted label vector, where n_samples in the number of samples.
    """
    print(crit)
    n_samples = X.shape[0]
    y_pred_set = np.zeros((n_samples, self._n_estimators)).astype(np.int8)
    distances_set = np.zeros((n_samples, self._n_estimators))
    confidence_score = np.zeros((n_samples, self._n_estimators))
    y_pred = np.zeros(n_samples).astype(np.int8)
    for i, model in enumerate(self._models):
      y_pred_set[:, i], confidence_score[:,i], distances_set[:, i] = model.predict(X, confidence_score = 1, distance_to_bmu = 1)
    if crit == 'max-voting-uniform':
      for i in range (n_samples):
        y_pred[i] = np.argmax(np.bincount(y_pred_set[i]))
    if crit == 'max-voting-weight':
      for i in range(n_samples):
        label_set = np.unique(y_pred_set[i])
        label_score = np.zeros(len(label_set))
        for j in range(self._n_estimators):
          if distances_set[i, j] == 0:
            label_score = np.zeros(len(label_set))
            label_score[label_set == y_pred_set[i, j]] = 1
            break
          label_score[label_set == y_pred_set[i, j]] += 1/distances_set[i, j]
        y_pred[i] = label_set[np.argmax(label_score)]  
    if crit == 'max-mean-confidence-score':
      for i in range(n_samples):
        label_set = np.unique(y_pred_set[i])
        label_score = np.zeros(len(label_set))
        for j in range(self._n_estimators):
          label_score[label_set == y_pred_set[i, j]] += confidence_score[i, j]
        y_pred[i] = label_set[np.argmax(label_score)]  
    return y_pred

class DistributionSomLvqNetworks(CombinationNetworksBase):
  """Distribution approach for combination of SOM-LVQ networks.
  
  Parameters
  ----------
  n_estimators : int
    Number of SOM-LVQ networks in combination.

  size_of_estimator : int
    Size of each SOM-LVQ networks.

  Attributes
  ----------
  _models : list
    List of models in combination.

  _features_set : list
    List of features using for each models.
  """
  def __init__(self, n_estimators, size_of_estimator):
    super().__init__(n_estimators, size_of_estimator)
    for _ in range (n_estimators):
      self._models.append(CombineSomLvq(n_rows = size_of_estimator, n_cols = size_of_estimator))

  def fit(self, X, y, features_selection = 'random', subset_size = 1,
          weights_init = None, labels_init = None,
          unsup_num_iters = 100, unsup_batch_size = 32,
          sup_num_iters = 100, sup_batch_size = 32,
          neighborhood = "bubble",
          learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = 0):
    """Fit the models according to the input data.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Training vectors, where n_samples is the number of samples and n_features is the number of features.

    y : 1D numpy array, shape (n_samples,)
      Training label vector, where n_samples in the number of samples.

    features_selection : str, options: ['random', 'weights']
      Options to select features.

    subset_size : float, default: 0.25
      Size of subset according to size of the dataset.

    weights_init : str, options: ['random', 'sample', 'pca'], default: random
      Strategies to initialize the weights of the neurons.

    labels_init : str, default: None
      Criteria to initialize the labels of the neurons.

    unsup_num_iters : int
      Number of iterations that unsupervised learning phase passes.

    unsup_batch_size : int
      Number of samples in the same batch during unsupervised learning phase,
      that uses the same learning rate and neighborhood's radius.

    sup_num_iters : int
      Number of iterations that supervised learning phase passes.

    sup_batch_size : int
      Number of samples in the same batch during supervised learning phase,
      that uses the same learning rate and neighborhood's radius.

    neighborhood : str, options: ['bubble', 'gaussian']
      Neighborhood function that is used to compute neighborhood coefficients.

    learning_rate : float, default: 0.5
      The learning rate that is used during training process.

    learning_decay_rate : float, default: 1
      The rate that is used to decrease the learning rate after each batches.

    learning_rate_decay_function : function, default: None
      The function that is used to decrease the learning rate after each batches.

    sigma : float, default: 1
      The radius that is used to compute neighborhood coefficients.

    sigma_decay_rate : float, default: 1
      The rate that is used to decrease the sigma after each batches.

    sigma_decay_function : function, default: None
      The function that is used to decrease the sigma after each batches.

    conscience : boolean, default: False
      The technique that tracks the frequency of winning of each neurons to avoid a neuron from winning too many times.

    verbose : boolean, optional, default False
      Verbose mode when fitting the models.

    Returns
    -------
    self : object
      Return self.
    """
    self._features_set = []
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    for _ in self._n_estimators:
      if features_selection == 'random':
        subset_features = np.random.randint(0, n_features, np.random.randint(1, n_features + 1))
        subset_features = np.unique(subset_features)
      elif features_selection == 'weights':
        corr_coef = np.corrcoef(np.append(X, y.reshape((-1, 1)), axis = 1).T)[-1, :-1]
        weights = np.abs(corr_coef.copy())
        subset_features = np.unique(weighted_sampling(weights, n_features))
      else:
        raise ValueError('features_selection should be random or weights')
      self._features_set.append(subset_features)
    
    for model in self._models:
      
      if subset_size != 1:
        subset_idx = np.random.randint(0, n_samples, int(subset_size * n_samples)).tolist()
      else:
        subset_idx = np.arange(n_samples)

      X_subset = X[subset_idx][:, subset_features]
      y_subset = y[subset_idx][:, subset_features]
      print(X_subset.shape)
      model.fit(X_subset, y_subset, weights_init = weights_init, labels_init = labels_init,
                unsup_num_iters = unsup_num_iters, unsup_batch_size = unsup_batch_size,
                sup_num_iters = sup_batch_size, sup_batch_size = sup_batch_size,
                neighborhood = neighborhood,
                learning_rate = learning_rate, learnining_decay_rate = learnining_decay_rate,
                learning_rate_decay_function = learning_rate_decay_function,
                sigma = sigma, sigma_decay_rate = sigma_decay_rate, sigma_decay_function = sigma_decay_function,
                conscience = conscience, verbose = verbose)

  def predict(self, X, crit = 'max-voting-uniform'):
    """Predict using the fitted model.
    
    Parameters
    ----------
    X : 2D numpy array, shape (n_samples, n_features)
      Input vectors, where n_samples is the number of samples and n_features is the number of features.

    crit : str, option: ['max-voting-uniform', 'max-voting-weight', 'max-mean-confidence-score']
      Criteria to combine results of models.

    Returns
    -------
    y_pred : 1D numpy array, shape (n_samples,)
      Predicted label vector, where n_samples in the number of samples.
    """
    print(crit)
    n_samples = X.shape[0]
    y_pred_set = np.zeros((n_samples, self._n_estimators)).astype(np.int8)
    distances_set = np.zeros((n_samples, self._n_estimators))
    confidence_score = np.zeros((n_samples, self._n_estimators))
    y_pred = np.zeros(n_samples).astype(np.int8)
    for i, model in enumerate(self._models):
      X_subset = X[:, self._features_set[i]]
      y_pred_set[:, i], confidence_score[:,i], distances_set[:, i] = model.predict(X_subset, confidence_score = 1, distance_to_bmu = 1)
    if crit == 'max-voting-uniform':
      for i in range (n_samples):
        y_pred[i] = np.argmax(np.bincount(y_pred_set[i]))
    if crit == 'max-voting-weight':
      for i in range(n_samples):
        label_set = np.unique(y_pred_set[i])
        label_score = np.zeros(len(label_set))
        for j in range(self._n_estimators):
          if distances_set[i, j] == 0:
            label_score = np.zeros(len(label_set))
            label_score[label_set == y_pred_set[i, j]] = 1
            break
          label_score[label_set == y_pred_set[i, j]] += 1/(distances_set[i, j]/len(self._features_set[j]))
        y_pred[i] = label_set[np.argmax(label_score)]  
    if crit == 'max-mean-confidence-score':
      for i in range(n_samples):
        label_set = np.unique(y_pred_set[i])
        label_score = np.zeros(len(label_set))
        for j in range(self._n_estimators):
          label_score[label_set == y_pred_set[i, j]] += confidence_score[i, j]
        y_pred[i] = label_set[np.argmax(label_score)]  
    return y_pred