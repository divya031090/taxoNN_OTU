import tensorflow as tf
import pandas as pd
from numpy import asarray, reshape
import T2D_input
import time
from sklearn.datasets import make_blobs
import ensembling_T2D
start_time = time.time()


def conv_layer(input_feat,conv_feature,x_shape,conv2d_stride,
  maxpool_ksize,maxpool_stride,in_ch):
  """ Convolutional layer """
  W_conv1 = weight_variable([1, input_feat, in_ch, conv_feature])
  b_conv1 = bias_variable([conv_feature])
  h_conv1 = tf.nn.relu(conv2d(x_shape, W_conv1,conv2d_stride) + b_conv1)
  #red_feat_dim1 = h_conv1.get_shape().as_list()[2]
  h_pool1 = max_pool_n(h_conv1,maxpool_ksize,maxpool_stride)
  # In order to get the reduced dimension of feature vector, we do following:
  # 2 is the index of the 4d tensor
  red_feat_dim1 = h_pool1.get_shape().as_list()[2]
  if (red_feat_dim1 < conv2d_stride):
    conv2d_stride = red_feat_dim1
  if (red_feat_dim1 < maxpool_stride):
    maxpool_stride = red_feat_dim1
  if (red_feat_dim1 < maxpool_ksize):
    maxpool_ksize = red_feat_dim1
  return [h_pool1,red_feat_dim1,conv2d_stride,maxpool_ksize,maxpool_stride,conv_feature]



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride_feat):
  return tf.nn.conv2d(x, W, strides=[1, 1, stride_feat, 1], padding='SAME')


def max_pool_n(x,maxp_k,maxp_str):
  return tf.nn.max_pool(x, ksize=[1, 1, maxp_k, 1],
                        strides=[1, 1, maxp_str, 1], padding='SAME')




## Input file specific variables

otuinfile = 'T2D_OTU.csv'
metadata = 'T2D_metadata.txt'
# Split 70% of data as training and 30% as test
train_ratio = 0.7
metavar = ['disease_stat','disease_status']  
levels = ['Positive','Negative']
# Read data
data = T2D_input.data_metadata_T2D(otuinfile,metadata,train_ratio,metavar,levels)
train_dataset = data[0]
test_input = data[1]
test_output = data[2]


# Data specific:
# No. of features

feature = 208
# No. of classification categories
resp = 2

# These variables are not having impact on accuracy
first_conv_feature = 32
second_conv_feature = 64
dense_layer_feature = 1024
opt_param = 1e-5
# For AdadeltaOptimizer()
#a = 0.1  # learning_rate
a=0.0001
b = 0.99  # rho
#c = 1e-8  # epsilon
c = 1e-5

X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
print(trainX.shape, testX.shape)

# load all models
n_members = 4

# Value to stride for conv2d
conv2d_stride = 1
# Value to stride and ksize for maxpool
maxpool_ksize = 1
maxpool_stride = 1
str_siz = [conv2d_stride,maxpool_ksize,maxpool_stride]

g=tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, [None, feature])
    y_ = tf.placeholder(tf.float32, [None, resp])

x_shape = tf.reshape(x, [-1,1,feature,1])


# Call convolutional layers
# The last argument, 1, is the value of first convolution input channel
with g.as_default():
    conv1 = conv_layer(feature,first_conv_feature,x_shape,
      str_siz[0],str_siz[1],str_siz[2],1)
    conv2 = conv_layer(conv1[1],second_conv_feature,conv1[0],
      conv1[2],conv1[3],conv1[4],conv1[5])
#conv2 = conv1[1],second_conv_feature,conv1[0],conv1[2],conv1[3],conv1[4],conv1[5]



# Densely Connected Layer
# Values of 23 * 1 * 64 come from printing (h_pool2)
# If ksize is 2, then feature value is 67 for densly connected layer
    W_fc1 = weight_variable([1 * conv2[1] * second_conv_feature, dense_layer_feature])
    b_fc1 = bias_variable([dense_layer_feature])
# Not doing pooling
    h_pool2_flat = tf.reshape(conv2[0], [-1, 1*conv2[1]*second_conv_feature]) #h_conv2
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
# Dropout or no drop-out has no impact
    #k_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, k_prob)

# Readout Layer
    W_fc2 = weight_variable([dense_layer_feature, resp])
    b_fc2 = bias_variable([resp])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2) #h_fc1_drop

#logits = tf.layers.dense(y_conv, 1, activation=None)

# Train and evaluate
with g.as_default():
    #cost = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits = y_conv,scope="Cost_function",weights=0.25)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
#print(cross_entropy)

#AdamOptimizer(opt_param)
with g.as_default():
    train_step = tf.train.AdadeltaOptimizer(a,b,c).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(a,name = "Optimizer").minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

#correct_prediction1 = tf.cast(tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)), tf.float32)
    
    correct_prediction1=tf.cast(correct_prediction, tf.float32)
#print("coreect_prediction is:",correct_prediction1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession(graph = g)
acc_ens=ensembling_T2D.ensemble(trainX, testX, trainy, testy,n_members)
init=tf.global_variables_initializer()
sess.run(init)
acc=0
i=0
#sess.run(tf.initialize_all_variables())
batch_xs_app=[]
batch_ys_app=[]
df3 = pd.DataFrame([])
df4 = pd.DataFrame([])
df_test = pd.DataFrame([])
#print("rows in train_dataset",train_dataset.count())
for index, row in train_dataset.iterrows():
  i=i+1
  # Store 0th-index is postive and 1st-index is negative
  #print("row in training dataset",row[1000])
  store = [0,0]
  otudat = row.drop('disease_status', axis=0).values
  #print("otudat is:",otudat)
  if row['disease_status'] == 'Positive':
    store[0] = 1
  else:
    store[1] = 1
  response = asarray(store)
  #print("response is:",response)
  batch_xs = reshape(otudat, (-1, 208))
  batch_xs_app.append(batch_xs)
  #print("train_data is:/n",batch_xs)
  batch_ys = reshape(response, (-1, 2))
  batch_ys_app.append(batch_ys)
  train_epochs=1
  for epoch in range(train_epochs):
     sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
  #print (sess.run(correct_prediction1, feed_dict={x: asarray(test_input), y_: asarray(test_output)}))
  pred2=correct_prediction1.eval({x: asarray(batch_xs), y_: asarray(batch_ys)})
  df2 = pd.DataFrame(pred2)
  df5 = pd.DataFrame(batch_ys)
  df3=df3.append(df2)
  df4 = df4.append(df5)
  #print("Accuracy:",accuracy.eval({x: asarray(batch_xs_app), y_: asarray(batch_ys_app)}))
  acc=acc+accuracy.eval({x: asarray(batch_xs), y_: asarray(batch_ys)})
i=i*0.28 
df3.to_csv("b2.csv",index=False)  
df4.to_csv("a2.csv",index=False)  
ensemb_acc=acc/i
df = pd.DataFrame(test_output)
df.to_csv("a.csv",index=False)
predi=correct_prediction1.eval({x: asarray(test_input), y_: asarray(test_output)})
df1 = pd.DataFrame(predi)
df_test= df_test.append(df1)
df_test.to_csv("b.csv",index=False)


print("Accuracy_ensemble taxoNN:",ensemb_acc)





