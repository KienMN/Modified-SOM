import numpy as np
import random

def weighted_sampling(weights, sample_size):
  """
  Returns a weighted sample with replacement.
  """
  totals = np.cumsum(weights)
  sample = []
  for _ in range(sample_size):
    rnd = random.random() * totals[-1]
    idx = np.searchsorted(totals, rnd, 'right')
    sample.append(idx)
  return sample