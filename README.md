# Image_Classifier
The repository consists of two Python Main programs:
i) train.py
ii) predict.py

Before developing the above command-line applications, the training & prediction implementation was tested on Jupyter notebook, 
the output of which can be accessed at 'Image Classifier Project.html'

The details of train & predict programs are described below.

train.py:

train.py trains a custom feed forward classifier built on torchvision's vgg16 CNN model using PyTorch. 

Once trained, the image classifier can take any flower image to predict the species of the same with more than 75% accuracy. 

The dataset for training, testing & validation of flower images has been taken from 
http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html that lists 102 categories of flowers.

The classifier can be trained on either GPU or CPU. 

User can set a number of parameters for training model such as learning rate, training epochs, 
number of hidden units etc.

At the end of training, the application prints out the training and validation loss along with validation accuracy.

predict.py:

predict.py reads the flower image and a saved training checkpoint to print the most likely image class and the associated probability.

User can set the number of classes to which associated probability for a particular image gets distributed.

The prediction can be made on either GPU or CPU.
