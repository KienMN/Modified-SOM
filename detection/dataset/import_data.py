import pandas as pd
import os

def load_dataset():
  filepath = os.path.dirname(__file__) + '/data/sample-well.csv'
  dataset = pd.read_csv(filepath)
  dataset = dataset[['DEPT', 'DT', 'GR', 'NPHI', 'PHIE', 'RHOB', 'VCL', 'DEPO_FACIES']]
  return dataset