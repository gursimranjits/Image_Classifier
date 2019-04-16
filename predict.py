# Imports here
import network
import utilities

def main_predict():
    image_path, topk, category_names, train_device, filepath, base_dir, architecture = utilities.command_line_parse_predict()
    trainloader, testloader, validloader, train_data = utilities.load_data(base_dir)
    trained_model = utilities.load_checkpoint(filepath, architecture, train_device)
    utilities.sanity_checking(image_path, trained_model, category_names, train_device, topk)

if __name__ == '__main__':
    main_predict()