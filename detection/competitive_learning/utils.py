def one_d_to_two_d_index(self, idx, n_rows, n_cols):
  return (idx // n_rows, idx % n_cols)

def two_d_to_one_d_index(self, x, y, n_rows):
  return x * self._n_rows + y

def default_learning_rate_decay_function(learning_rate_0, learning_decay_rate, batch):
  return (learning_rate_0) / (1 + batch * learning_decay_rate)

def default_sigma_decay_function(sigma_0, sigma_decay_rate, batch):
  return (sigma_0) / (1 + batch * sigma_decay_rate)