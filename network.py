# Imports here
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


#Building and training the network
def neuralnet_setup(architecture = 'vgg16', dropout = '0.5', lr = '0.001', hidden_layer1 = 6000):
    '''
    Argument: The architecture for the network(alexnet, vgg16), the hyperparameters for the network (hidden layer 1 nodes,
    dropout and learning rate)
    Return: The set up model, along with the criterion and the optimizer for the Training
    '''
    if architecture ==  'vgg16':
        model = models.vgg16(pretrained=True)
    elif architeture == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
         print("I'm sorry but {} is not a valid model.\n Please enter vgg16 or alexnet".format(structure))
         raise Exception("Please Try Again!")

    for param in model.parameters():
               param.requires_grad = False
    input_layer   = 25088
    hidden_layer2 = 1000
    output_layer  = 102

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_layer, hidden_layer1)),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                                            ('relu2', nn.ReLU()),
                                            ('logits', nn.Linear(hidden_layer2, output_layer)),
                                            ('output', nn.LogSoftmax(dim=1)),
                                            ('dropout', nn.Dropout(0.5))
                                           ]))
    model.classifier  = classifier

    #Defining the criterion and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    return model, criterion, optimizer

#Network validation function

def validation(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0
    for ii, (inputs_valid, labels_valid) in enumerate(validloader):
        inputs_valid, labels_valid = inputs_valid.to(device), labels_valid.to(device)
        outputs_valid = model.forward(inputs_valid)
        valid_loss += criterion(outputs_valid, labels_valid).item()
        ps_valid = torch.exp(outputs_valid)
        equality = (labels_valid.data == ps_valid.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

def neuralnet_train(model, criterion, optimizer, trainloader, validloader, testloader, epochs = 5, print_every=50, train_device='GPU'):
               '''
               Argument: The model, criterion, optimizer, number of epochs, training dataset and option of training over GPU
               Return: Nothing
               This function trains the model over a certain number of epochs and displays the training,validation and accuracy every
               "print_every" step using cuda if specified.
               The training method is specified by the criterion and the optimizer.
               '''
               #Architecture Agnostic Torch Device
               if((train_device == 'GPU') and (torch.cuda.is_available())):
                   device = torch.device("cuda:0")
                   print("--------------Training on GPU--------------")
               else:
                   device = torch.device("cpu")
                   print("--------------GPU is not available, executing on CPU-------------")
               steps = 0
               model.to(device)

               print("--------------Training started-------------")
               for e in range(epochs):
                   running_loss = 0
                   model.train()
                   for ii, (inputs, labels) in enumerate(trainloader):
                       steps += 1
                       inputs, labels = inputs.to(device), labels.to(device)
                       optimizer.zero_grad()
                       outputs = model.forward(inputs)
                       loss = criterion(outputs, labels)
                       loss.backward()
                       optimizer.step()

                       running_loss += loss.item()

                       if steps % print_every == 0:
                           model.eval()
                           with torch.no_grad():
                               valid_loss, accuracy = validation(model, validloader, criterion, device)
                           print("Epoch: {}/{}.. ".format(e+1, epochs),
                           "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                           "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                           "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                           running_loss = 0
                           model.train()
               # Validation using the test set
               print("--------------Validation using test set started------------- ")
               model.eval()
               test_loss = 0
               accuracy = 0
               for ii, (inputs_test, labels_test) in enumerate(testloader):
                   inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                   with torch.no_grad():
                       output_test = model.forward(inputs_test)
                   test_loss += criterion(output_test, labels_test).item()
                   ps_test = torch.exp(output_test)
                   equality = (labels_test.data == ps_test.max(dim=1)[1])
                   accuracy += equality.type(torch.FloatTensor).mean()
                   print("Loss of Test Set: {:.3f}.. ".format(test_loss/len(testloader)),
                   "Accuracy of Test Set: {:.3f}".format(accuracy/len(testloader)))

               print("-------------- Training finished!-----------------------")
               print("---------------Epochs: {}-------------------------------".format(epochs))
               print("---------------Steps: {}--------------------------------".format(steps))
               return model
