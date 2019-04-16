#Imports here
import network
import utilities

def main_train():
    base_dir, save_path, lr, architecture, dropout, hidden_layer1, train_device, epochs, print_every = utilities.command_line_parse_train()
    trainloader, testloader, validloader, train_data = utilities.load_data(base_dir)
    model, criterion, optimizer = network.neuralnet_setup(architecture, dropout, lr, hidden_layer1)
    trained_model = network.neuralnet_train(model, criterion, optimizer, trainloader, validloader,
                                            testloader, epochs, print_every, train_device)
    utilities.save_checkpoint(trained_model, train_data, save_path)
    print("---------------Finished Model Training & Saving of checkpoint!--------------")

if __name__ == '__main__':
    main_train()