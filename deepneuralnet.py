import tflearn
# to count folders, for final layer
import os 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()

###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 1],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 64, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 128, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 128, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 2048, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, .5)

# 8: Fully-connected layer (needs to know number of letters)
numLetters = 0

for _, dirnames, filenames in os.walk("/home/lenny/Desktop/AI_Proj/Code/Samples/Train/"):
  # ^ this idiom means "we won't be using this value"
    numLetters += len(dirnames)

network = fully_connected(network, numLetters, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy', learning_rate=0.0005, metric=acc)
# The model with details on where to save
# Will save in current directory
model = tflearn.DNN(network)#tensorboard_verbose=3)
