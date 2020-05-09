import numpy as np
import pandas as pd
import math


def data_metadata_T2D(otuinfile,metadata,train_ratio,metavar,levels):
  """ Reads OTU table data and meta data and creates train/test dataset """
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
  tr = 0.5
  v = [levels[0] if x == 'TRUE' else levels[1] for x in u]
  b.loc[:,metavar[1]] = pd.Series(v, index=b.index)
  c = b[b[metavar[1]].isin([levels[0], levels[1]])]
  #print("a is",a[1600:1615])
  #print("b is",b[1600:1615])
  #print("c is",c[1600:1615])
  #c = b
  # No. of samples to train/test the model
  
  n_train = int(math.ceil(tr*c.shape[0]))
  #n_train = round(n_train/2)
  
  #n_train = int(math.ceil(train_ratio*randperm))
  #idx = np.random.randint(c.shape[0], size=n_train)
  #print(idx)
  train_dataset1 = pd.DataFrame()
  test_dataset = pd.DataFrame()
  train_dataset1 = c[:n_train]
  test_dataset1 = c[n_train:]
  

  
  test_dataset=test_dataset1.sample(frac=1)
  test_input = []
  
  #test_input = test_dataset
  test_output = []
  
  for index, row in test_dataset.iterrows():
    # Store 0th-index is  postive and 1st-index is negative
    store = [0,0]
    otudat = row.drop(metavar[1], axis=0).values
    #print("otudatIs:",otudat)
    if row[metavar[1]] == levels[0]:
      store[0] = 1
    else:
      store[1] = 1
    
    
    test_input.append(otudat)
    test_output.append(store)
    
  
  train_dataset=train_dataset1.sample(frac=1)
  
  print("test_dataset Is:",test_dataset) 
  print("train_dataset Is:",train_dataset) 
  print("test_output is:",test_output)
  return [train_dataset, test_input, test_output]
  


