# Testing combination of som-lvq models

# Importing the context
import context

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the random state
random_state = 17

# Importing the dataset
from detection.dataset import load_dataset
dataset = load_dataset()
X = dataset.iloc[:, 1: -1].values
y = dataset.iloc[:, -1].values.astype(np.int8)

# Spliting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state=random_state)

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

print('=' * 17, 'CombinationSomLvqNetworks', '=' * 17)
model = CombinationSomLvqNetworks(n_estimators=10, size_of_estimator=3,
                                  random_state=random_state)
model.fit(X_train, y_train, weights_init=None, labels_init=None, 
          unsup_num_iters=10, unsup_batch_size=10, sup_num_iters=10, 
          sup_batch_size=10, neighborhood="bubble", learning_rate=0.5, 
          learning_decay_rate=1, learning_rate_decay_function=None, sigma=1, 
          sigma_decay_rate=1, sigma_decay_function=None, conscience=False, 
          verbose=0)

y_pred = model.predict(X_test, crit = 'max-voting-weight')
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print('Accuracy:', accuracy_score(y_test, y_pred))

model_clone = CombinationSomLvqNetworks(n_estimators=10, size_of_estimator=3,
                                        random_state=random_state)
model_clone.fit(X_train, y_train, weights_init=None, labels_init=None, 
                unsup_num_iters=10, unsup_batch_size=10, sup_num_iters=10, 
                sup_batch_size=10, neighborhood="bubble", learning_rate=0.5, 
                learning_decay_rate=1, learning_rate_decay_function=None, sigma=1, 
                sigma_decay_rate=1, sigma_decay_function=None, conscience=False, 
                verbose=0)

y_pred_clone = model_clone.predict(X_test, crit = 'max-voting-weight')
y_pred_clone = encoder.inverse_transform(y_pred_clone)

print('Reproducable', (y_pred == y_pred_clone).all())


print('=' * 17, 'DistributionSomLvqNetworks', '=' * 17)
model = DistributionSomLvqNetworks(n_estimators=10, size_of_estimator=3,
                                   random_state=random_state)
model.fit(X_train, y_train, features_selection='weights', weights_init=None, 
          labels_init=None, unsup_num_iters=10, unsup_batch_size=10,
          sup_num_iters=10, sup_batch_size=10, neighborhood = "bubble",
          learning_rate=0.5, learning_decay_rate=1, 
          learning_rate_decay_function=None, sigma=1, sigma_decay_rate=1, 
          sigma_decay_function=None, conscience=False, verbose=0)

y_pred = model.predict(X_test, crit = 'max-voting-weight')
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
print('Accuracy:', accuracy_score(y_test, y_pred))

model_clone = DistributionSomLvqNetworks(n_estimators=10, size_of_estimator=3,
                                         random_state=random_state)
model_clone.fit(X_train, y_train, features_selection='weights', weights_init=None, 
                labels_init=None, unsup_num_iters=10, unsup_batch_size=10,
                sup_num_iters=10, sup_batch_size=10, neighborhood = "bubble",
                learning_rate=0.5, learning_decay_rate=1, 
                learning_rate_decay_function=None, sigma=1, sigma_decay_rate=1, 
                sigma_decay_function=None, conscience=False, verbose=0)

y_pred_clone = model_clone.predict(X_test, crit = 'max-voting-weight')
y_pred_clone = encoder.inverse_transform(y_pred_clone)

print('Reproducable', (y_pred == y_pred_clone).all())