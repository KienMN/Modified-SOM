# Testing single som and lvq network
import context

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
from detection.competitive_learning import CombineSomLvq
model = CombineSomLvq(n_rows = 3, n_cols = 3)
model.fit(X_train, y_train, weights_init = "pca", labels_init = None,
          unsup_num_iters = 50, unsup_batch_size = 10,
          sup_num_iters = 50, sup_batch_size = 10,
          neighborhood = "gaussian",
          learning_rate = 0.75, learning_decay_rate = 1, learning_rate_decay_function = None,
          sigma = 1, sigma_decay_rate = 1, sigma_decay_function = None,
          conscience = False, verbose = 0)

model.fit(X_train, y_train, weights_init = "pca", labels_init = None,
          unsup_num_iters = 50, unsup_batch_size = 10,
          sup_num_iters = 50, sup_batch_size = 10,
          neighborhood = "gaussian",
          learning_rate = None, learning_decay_rate = None, learning_rate_decay_function = None,
          sigma = None, sigma_decay_rate = None, sigma_decay_function = None,
          conscience = False, verbose = 0)

# Predict the result
y_pred = model.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix
print(cm)
true_result = 0
for i in range (len(cm)):
  true_result += cm[i][i]
print('Accuracy:', true_result / np.sum(cm))