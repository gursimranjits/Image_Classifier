# Imports here
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import argparse

#Argument parsing for prediction functionality
def command_line_parse_predict():
    argp = argparse.ArgumentParser(description='predict.py')
    #Command Line Arguments
    argp.add_argument('--input_image', dest='input_image', default='./flowers/test/15/image_06360.jpg',
                      action="store", type = str)
    argp.add_argument('--load_checkpoint', dest='load_checkpoint', default='./checkpoint.pth',
                      action="store", type = str)
    argp.add_argument('--top_k', dest="top_k", default=5, action="store", type=int)
    argp.add_argument('--category_names', dest="category_names", default='cat_to_name.json',
                      action="store")
    argp.add_argument('--prediction_device', dest="prediction_device", default="GPU", help="Choose between GPU or cpu as prediction device", action="store")
    parg = argp.parse_args()
    image_path = parg.input_image
    topk = parg.top_k
    category_names = parg.category_names
    train_device = parg.prediction_device
    filepath = parg.load_checkpoint
    base_dir = image_path.split('/')[1]
    architecture = 'vgg16'

    return image_path, topk, category_names, train_device, filepath, base_dir, architecture

#Command line parsing for training functionality
def command_line_parse_train():
    argp = argparse.ArgumentParser(description='train.py')
    
    # Command Line ardguments
    argp.add_argument('--base_dir', dest='base_dir', action="store", default="./flowers/")
    argp.add_argument('--train_device', dest="train_device", action="store", default="GPU", help="Choose between GPU or cpu as training device")
    argp.add_argument('--save_path', dest="save_path", action="store", default="./checkpoint.pth")
    argp.add_argument('--lr', dest="lr", action="store", default=0.001)
    argp.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
    argp.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
    argp.add_argument('--architecture', dest="architecture", action="store", default="vgg16", type = str)
    argp.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=6000)
    
    parg = argp.parse_args()
    base_dir = parg.base_dir
    save_path = parg.save_path
    lr = parg.lr
    architecture = parg.architecture
    dropout = parg.dropout
    hidden_layer1 = parg.hidden_units
    train_device = parg.train_device
    epochs = parg.epochs
    print_every = 50

    return base_dir, save_path, lr, architecture, dropout, hidden_layer1, train_device, epochs, print_every

#Neural Net architecture
arch = {"vgg16":25088,
       "alexnet": 8329}

#Data Load Function
def load_data(base_dir = "./flowers/"):
      '''
      Argument : the dataset's path
      Return : The loaders for the training, validation and test datasets
      This function receives the location of the image files, applies the necessery transformations
      (rotations,flips,normalizations and
      crops) and converts the images to tensors in order to be able to fed into the neural network
      '''
      #Data sources
      data_dir = base_dir
      train_dir = data_dir + '/train'
      valid_dir = data_dir + '/valid'
      test_dir = data_dir + '/test'
      # Transforms definition for the training, validation, and testing sets
      train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    
      test_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
        
      valid_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
      #Loading the datasets with ImageFolder
      train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
      test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
      valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
      # Defining the dataloaders using the image datasets and the transforms
      trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
      testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
      validloader = torch.utils.data.DataLoader(test_data, batch_size=24)
    
      return trainloader, testloader, validloader, train_data

# Saving the checkpoint
def save_checkpoint(model, train_data, save_path='checkpoint.pth'):
      '''
      Argument: The model and the desired path of the checkpoint
      Return: Nothing
      This function saves the model at a location specified by the user path
      '''
      model.class_to_idx = train_data.class_to_idx
      checkpoint = {'classifier': model.classifier,
      'state_dict': model.state_dict(),
      'class_to_idx': model.class_to_idx}

      torch.save(checkpoint, save_path)

# Function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath, architecture, train_device):
    '''
    Argument: The path of the checkpoint file
    Return: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    if((train_device == 'GPU') and (torch.cuda.is_available())):
        checkpoint = torch.load(filepath)           
    else:                 
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        
    if architecture ==  'vgg16':
        model = models.vgg16(pretrained=True)
    elif architeture == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("I'm sorry but {} is not a valid model.\n Please enter vgg16 or alexnet".format(structure))
        raise Exception("Please Try Again!")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

#Process a PIL image for use in a PyTorch model
def process_image(image):
    '''
    Argument: PLI Image
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    Return: Processed Image
    '''
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256,10000))
    left_corner = (image.width- 224)/2
    right_corner = left_corner + 224
    top_corner = (image.height - 224)/2
    bottom_corner = top_corner + 224
    image = image.crop((left_corner, top_corner, right_corner, bottom_corner))

    np_image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])

    processed_image = (np_image - mean)/std_dev
    processed_image  = processed_image.transpose((2, 0, 1))

    return processed_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# Implementing the code to predict the class from an image file

def predict(image_path, model, train_device, cat_to_name, topk=5):
    '''
    Argument: Image Path, Model, topk
    Return: Top probabilities, Top labels
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if((train_device == 'GPU') and (torch.cuda.is_available())):
        device = torch.device("cuda:0")
        print("--------------Executing on GPU--------------")
    else:
        device = torch.device("cpu")
        print("--------------GPU is not available, executing on CPU-------------")
    model.to(device)

    image = Image.open(image_path)

    processed_image = process_image(image)


    processed_tensor_image = torch.from_numpy(processed_image).type(torch.FloatTensor)
    processed_tensor_image = processed_tensor_image.unsqueeze_(0)
    tensor_image = processed_tensor_image.to(device)

    with torch.no_grad():
        output = model.forward(tensor_image)

    ps = F.softmax(output.data, dim = 1)

    top_ps,top_lbs = ps.topk(topk)
    top_ps = top_ps.cpu().numpy()
    top_lbs = top_lbs.cpu().numpy().tolist()[0]

    idx_to_class = {val: key for key,val in model.class_to_idx.items()}

    top_labels = [cat_to_name[idx_to_class[lbs]] for lbs in top_lbs]

    return top_ps, top_labels

# Display an image along with the top 5 classes

def sanity_checking(image_path, model, category_names, train_device, topk):
    '''
    Argument: Image Path, Model
    Return: Nothing
    Checks the sanity of prediction
    '''
    #For Label Mapping
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    idx = image_path.split('/')[3]
    header = cat_to_name[idx]
    image = Image.open(image_path)
    processed_image = process_image(image)
    ps, lbs = predict(image_path, model, train_device, cat_to_name, topk)
    print("\n--------------Actual Label of the image is {}-------------\n".format(header))
    print("--------------Top {} classes predicted by NeuralNet are:------------- ".format(topk))
    for i in range(topk):
        print("Predicted Label: {:20}    Predicted Probability:{:.3f}".format(lbs[i], ps[0][i]))