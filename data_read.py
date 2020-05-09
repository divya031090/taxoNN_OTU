import numpy as np
import pandas as pd
import math


def load():
    ## Input file specific variables
  otuinfile = 'Sim_OTU.csv'
  metadata = 'Sim_metadata.txt'
# Split 70% of data as training and 30% as test
  train_ratio = 0.3
  metavar = ['disease_stat','disease_status']  
  levels = ['Positive','Negative']
  a = pd.read_table(otuinfile,skiprows=1,index_col=0)
  #a = pd.read_table(otuinfile,skiprows=1,index_col=0).iloc[:, 0:]
  #b = a.transpose()
  b = a
  #print("b is:",b)
  response = {}
  var = 0
  infile = open(metadata,'rU')
  for line in infile:
    if line.startswith("#SampleID"):
      spline = line.strip().split("\t")
      var = spline.index(metavar[0])
    if not line.startswith("#SampleID"):
      spline = line.strip().split("\t")
      response[spline[0]] = spline[var]
  u = [response[x] for x in list(b.index)]
  #print(u)
  v = [levels[0] if x == 'TRUE' else levels[1] for x in u]
  b.loc[:,metavar[1]] = pd.Series(v, index=b.index)
  c = b[b[metavar[1]].isin([levels[0], levels[1]])]
  #print("a is",a[1600:1615])
  #print("b is",b[1600:1615])
  #print("c is",c[1600:1615])
  #c = b
  # No. of samples to train/test the model
  n_train = int(math.ceil(train_ratio*c.shape[0]))
  print("n_train",n_train)
  train_dataset = pd.DataFrame()
  test_dataset = pd.DataFrame()
  train_dataset = c[:n_train]
  test_dataset = c[n_train:]
  #test_dataset = c[n_train:(n_train*2)]
  #test_dataset = c[(n_train*2):]
  #print("train_dataset Is:",train_dataset)
  #print("train_dataset Is:",train_dataset[1600:1615])
  test_dataset=test_dataset.sample(frac=1)
  train_dataset=train_dataset.sample(frac=1)
  return [train_dataset.drop(metavar[1],1),
  train_dataset[[metavar[1]]],
  test_dataset.drop(metavar[1],1),
  test_dataset[[metavar[1]]]]
