"""CycleTransformer: Text-to-Image-to-Text Using Cycle Consistency
Matan Kleiner & Hila Manor
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import argparse
import yaml
import training
import utils
import torch
from FlowersDataset import ImageCaption102FlowersDataset
from torchvision import transforms

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    # ----- Creating Argument Parser -----
    parser = argparse.ArgumentParser('CycleTransformer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The amount of epochs to train the model')
    parser.add_argument('--val_epochs', type=int, default=5,
                        help='The amount of epochs for which to check the validation metrics')
    parser.add_argument('--config', default='../configs/config.yaml', type=str,
                        help='Path to YAML config file. Default: config.yaml')
    parsed_args = parser.parse_args()

    # This type of loading gives precedence to the parser arguments
    with open(parsed_args.config) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args.update(vars(parsed_args))

    # set seed for reproducibility
    utils.set_seed(42)

    # create dataset
    if args['db_type'] == 'flowers':
        transformations = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    utils.create_output_dir(args)

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    training.train(args, dataset, device)
