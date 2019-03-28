import numpy as np
from .base import CompetitiveNetwork
from .utils import weights_initialize

class TheoriticalLvq(CompetitiveNetwork):
  """Learning vector quantization"""
  def __init__(self):
    pass

  def fit(self, X, y, labels_proportion = None):
    pass

  def predict(self):
    pass

class NeighborhoodLvq(CompetitiveNetwork):
  """Modified version of theoretical LVQ"""
  def __init__(self):
    pass

  def fit(self, X, y, labels_init = None):
    pass

  def predict(self, X):
    pass