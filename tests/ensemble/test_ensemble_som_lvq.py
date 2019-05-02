# Testing combination of som-lvq models

# Importing the context
import context

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest

# Importing the dataset
from detection.dataset import load_dataset
dataset = load_dataset()
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# Training the LVQ
from detection.ensemble import CombinationSomLvqNetworks
from detection.ensemble import DistributionSomLvqNetworks

class TestCombinationSomLvqNetworks(unittest.TestCase):
  def test_creating(self):
    n_estimators = 10
    model = CombinationSomLvqNetworks(n_estimators = n_estimators, size_of_estimator = 3)
    self.assertEqual(len(model._models), n_estimators)
  
  def test_training(self):
    n_estimators = 10
    size = 3
    model = CombinationSomLvqNetworks(n_estimators = n_estimators, size_of_estimator = size)
    model.fit(X_train, y_train, weights_init = None, labels_init = None,
              unsup_num_iters = 10, unsup_batch_size = 10,
              sup_num_iters = 10, sup_batch_size = 10,
              neighborhood = "bubble",
              learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
              sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
              conscience = False, verbose = 0)
    self.assertEqual(model._models[0]._competitive_layer_weights.shape, (size * size, X_train.shape[1]))

  def test_predicting(self):
    n_estimators = 10
    size = 3
    model = CombinationSomLvqNetworks(n_estimators = n_estimators, size_of_estimator = size)
    model.fit(X_train, y_train, subset_size = 0.25, weights_init = None, labels_init = None,
              unsup_num_iters = 10, unsup_batch_size = 10,
              sup_num_iters = 10, sup_batch_size = 10,
              neighborhood = "bubble",
              learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
              sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
              conscience = False, verbose = 0)
    y_pred = model.predict(X_test, crit = 'max-voting-weight')
    y_pred = encoder.inverse_transform(y_pred)
    self.assertEqual(y_pred.shape, y_test.shape)

class TestDistributionSomLvqNetworks(unittest.TestCase):
  def test_creating(self):
    n_estimators = 10
    model = DistributionSomLvqNetworks(n_estimators = n_estimators, size_of_estimator = 3)
    self.assertEqual(len(model._models), n_estimators)
  
  def test_training(self):
    n_estimators = 10
    size = 3
    model = DistributionSomLvqNetworks(n_estimators = n_estimators, size_of_estimator = size)
    model.fit(X_train, y_train, features_selection = 'weights', weights_init = None, labels_init = None,
              unsup_num_iters = 10, unsup_batch_size = 10,
              sup_num_iters = 10, sup_batch_size = 10,
              neighborhood = "bubble",
              learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
              sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
              conscience = False, verbose = 0)
    self.assertEqual(model._models[0]._competitive_layer_weights.shape, (size * size, len(model._features_set[0])))

  def test_predicting(self):
    n_estimators = 10
    size = 3
    model = DistributionSomLvqNetworks(n_estimators = n_estimators, size_of_estimator = size)
    model.fit(X_train, y_train, features_selection = 'weights', weights_init = None, labels_init = None,
              unsup_num_iters = 10, unsup_batch_size = 10,
              sup_num_iters = 10, sup_batch_size = 10,
              neighborhood = "bubble",
              learning_rate = 0.5, learnining_decay_rate = 1, learning_rate_decay_function = None,
              sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
              conscience = False, verbose = 0)
    y_pred = model.predict(X_test, crit = 'max-voting-weight')
    y_pred = encoder.inverse_transform(y_pred)
    self.assertEqual(y_pred.shape, y_test.shape)

if __name__ == '__main__':
  unittest.main()