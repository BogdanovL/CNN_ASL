# predict.py
import random 
import importlib
from tflearn.data_utils import image_preloader as pl
from time import sleep

import deepneuralnet as net 
model = net.model
path_to_model = 'final-model1.tflearn'
model.load(path_to_model)

alphabet = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
for item in alphabet:
	# load spectrogram
	path = "/home/lenny/Desktop/AI_Proj/Code/TestFilesGoHere/" + item
	testX, testY = pl(path, image_shape=(64, 64), mode='folder', categorical_labels=True, normalize=True)         

	# Surely there's a better way to pass one image than this ***hack!***
	x = testX[0].reshape((64, 64, 1))

	# Get model prediction
	result = model.predict([x])[0]

	# Interpret result from model to get user and percentage
	prediction = result.tolist().index(max(result)) # The index represents the number predicted in this case
	print("Got Prediction!")
	# Write to output file
	lookup = dict({0:"A", 
		       1:"B", 
		       2:"C", 
		       3:"D", 
			4:"E", 
			5:"F", 
			6:"G", 
			7:"H", 
			8:"I", 
			9:"K", 
			10:"L", 
			11:"M", 
			12:"N", 
			13:"O", 
			14:"P", 
			15:"Q", 
			16:"R", 
			17:"S", 
			18:"T", 
			19:"U", 
			20:"V", 
			21:"W", 
			22:"X", 
			23:"Y" })
	resultLetter = lookup[prediction]
	file = open("/home/lenny/Desktop/AI_Proj/Results/report_" + item + ".txt", "w")
	file.write("{p} {perc}\n".format(p=resultLetter, perc=max(result)))
	print("{p} {perc}".format(p=resultLetter, perc=max(result)))
	for x in range (0, 24):
		print("{p} with {perc}%" .format(p=lookup[x], perc=round(result[x] * 100, 2)))
		file.write("{p} with {perc}%\n" .format(p=lookup[x], perc=round(result[x] * 100, 5)))
	file.close()
	

