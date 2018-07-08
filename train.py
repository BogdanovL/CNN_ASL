# train.py
import deepneuralnet as net
import numpy as np
from tflearn.data_utils import image_preloader as pl
# Get the model
model = net.model

# Load data
# Training Data
testX, testY = pl("/home/lenny/Desktop/AI_Proj/Code/Samples/Test", image_shape=(64, 64, 1), mode='folder', categorical_labels=True, normalize=True)            
# Validiation Data
X, Y = pl("/home/lenny/Desktop/AI_Proj/Code/Samples/Train", image_shape=(64, 64, 1), mode='folder', categorical_labels=True, normalize=True)
X = np.reshape(X, (-1, 64, 64, 1))
testX = np.reshape(testX, (-1, 64, 64, 1))

#model.load("final-model1.tflearn")
model.fit(X, Y, n_epoch=2, validation_set=(testX, testY), show_metric=True, run_id="deep_nn")
model.save('final-model1.tflearn')

#file2 = open('/home/lenny/ModelData/threadCom.txt', 'w+')
#file2.write("TRAINING DONE")
#file2.close()
