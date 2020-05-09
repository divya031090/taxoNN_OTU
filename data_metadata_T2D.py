import numpy as np
import pandas as pd
import math


def load():
    ## Input file specific variables
  otuinfile = 'T2D_OTU.csv'
  metadata = 'T2D_metadata.txt'

  metavar = ['disease_stat','disease_status']  
  levels = ['Positive','Negative']
  a = pd.read_table(otuinfile,skiprows=1,index_col=0)
  #a = pd.read_table(otuinfile,skiprows=1,index_col=0).iloc[:, 0:]
  #b = a.transpose()
  b = a
  #print("b is:",b)
  response = {}
  hiv = 0
  infile = open(metadata,'rU')
  for line in infile:
    if line.startswith("#SampleID"):
      spline = line.strip().split("\t")
      hiv = spline.index(metavar[0])
    if not line.startswith("#SampleID"):
      spline = line.strip().split("\t")
      response[spline[0]] = spline[hiv]
  u = [response[x] for x in list(b.index)]
  #print(u)
  v = [levels[0] if x == 'TRUE' else levels[1] for x in u]
  b.loc[:,metavar[1]] = pd.Series(v, index=b.index)
  c = b[b[metavar[1]].isin([levels[0], levels[1]])]
  
  # No. of samples to train/test the model
  n_train = int(math.ceil(train_ratio*c.shape[0]))
  print("n_train",n_train)
  train_dataset = pd.DataFrame()
  test_dataset = pd.DataFrame()
  train_dataset = c[:n_train]
  test_dataset = c[n_train:]
  
  test_dataset=test_dataset.sample(frac=1)
  train_dataset=train_dataset.sample(frac=1)
  return [train_dataset.drop(metavar[1],1),
  train_dataset[[metavar[1]]],
  test_dataset.drop(metavar[1],1),
  test_dataset[[metavar[1]]]]


