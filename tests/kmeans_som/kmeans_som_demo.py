# Testing single som and lvq network
import context

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Label encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)

# Training the LVQ
from detection.competitive_learning import SOM
model = SOM(n_rows=3, n_cols=3, random_state=random_state)
model.fit(X_train, weights_init= "pca",num_iters=100, batch_size=32, 
          neighborhood="gaussian", learning_rate=0.75, learning_decay_rate=1, 
          learning_rate_decay_function=None, sigma=1, sigma_decay_rate=1, 
          sigma_decay_function=None, conscience=False, num_clusters=4, 
          verbose=0)


# Predict the result
pred = model.predict(X_train)
print(pred)

model_clone = SOM(n_rows=3, n_cols=3, random_state=random_state)
model_clone.fit(X_train, weights_init= "pca",num_iters=100, batch_size=32, 
                neighborhood="gaussian", learning_rate=0.75, learning_decay_rate=1, 
                learning_rate_decay_function=None, sigma=1, sigma_decay_rate=1, 
                sigma_decay_function=None, conscience=False, num_clusters=4, 
                verbose=0)
pred_clone = model_clone.predict(X_train)
print('Reproducable:', (pred == pred_clone).all())