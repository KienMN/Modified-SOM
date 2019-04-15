import numpy as np
from ..competitive_learning.som import CombineSomLvq

class CombinationSomLvqNetworks:
  def __init__(self, n_estimators, size_of_estimator):
    self._n_estimators = n_estimators
    self._models = []
    for _ in range (n_estimators):
      self._models.append(CombineSomLvq(n_rows = size_of_estimator, n_cols = size_of_estimator))

  def fit(self, X, y, subset_size = 0.25, weights_init = None, labels_init = None,
          unsup_num_iters = 100, unsup_batch_size = 32,
          sup_num_iters = 100, sup_batch_size = 32,
          neighborhood = "bubble",
          learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = 0):
    n_samples = X.shape[0]
    for model in self._models:
      subset_idx = np.random.randint(0, n_samples, int(subset_size * n_samples))
      X_subset = X[subset_idx]
      y_subset = y[subset_idx]
      model.fit(X_subset, y_subset, weights_init = None, labels_init = None,
                unsup_num_iters = unsup_num_iters, unsup_batch_size = unsup_batch_size,
                sup_num_iters = sup_batch_size, sup_batch_size = sup_batch_size,
                neighborhood = neighborhood,
                learning_rate = learning_rate, learnining_decay_rate = learnining_decay_rate,
                learning_rate_decay_function = learning_rate_decay_function,
                sigma = sigma, sigma_decay_rate = sigma_decay_rate, sigma_decay_function = sigma_decay_function,
                conscience = conscience, verbose = verbose)

  def predict(self, X, crit = 'max-voting-uniform'):
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