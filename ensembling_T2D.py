from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax

 

 
# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(10, activation='relu')(merge)
	output = Dense(3, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
#	plot_model(model, show_shapes=True, to_file='model_graph.png')
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
 
# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'CNN' + str(i + 1) + '_T2D'
		file = 'model_' + str(i + 1) + '_T2D.h5'
		# load model from file
		model = load_model(file)
		# add to list of members
		all_models.append(model)
		print('%s' % filename)
	return all_models
 
# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# encode output data
	inputy_enc = to_categorical(inputy)
	# fit model
	model.fit(X, inputy_enc, epochs=400, verbose=0)
 
# make a prediction with our stacked model on all the phyla
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)
 
def ensemble(trainX, testX, trainy, testy,n_members):
    members = load_all_models(n_members)
    print('%d phyla training' % len(members))
    # define ensemble model
    stacked_model = define_stacked_model(members)
    # fit stacked model on test dataset
    fit_stacked_model(stacked_model, testX, testy)
    # make predictions and evaluate
    yhat = predict_stacked_model(stacked_model, testX)
    yhat = argmax(yhat, axis=1)
    acc = accuracy_score(testy, yhat)
    return acc