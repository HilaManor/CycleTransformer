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
import os

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
    parser.add_argument('--baseline', action='store_true',
                        help='Train the baseline models instead of the cycle consistent model')
    parser.add_argument('--continue_training', type=str, default=None,
                        help='Continue to train the model from past saved models at the given folder')
    parsed_args = parser.parse_args()

    # This type of loading gives precedence to the parser arguments
    with open(parsed_args.config) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args.update(vars(parsed_args))

    # If the option for training continuation from a checkpoint was marked, check if it's possible to continue
    if args["continue_training"] is not None:
        # The first saved thing is the generator_k1.pth file, so if no such file exists - there is no mid-training
        # state to continue from
        saved_gen_path = os.path.join(args["continue_training"], f'generator_k{1}.pth')
        if os.path.exists(saved_gen_path):
            args = torch.load(saved_gen_path)['args']
            args["continue_training"] = True
        else:
            args["continue_training"] = False

    # set seed for reproducibility
    utils.set_seed(42)

    # create dataset
    if args['db_type'] == 'flowers':
        transformations = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    if not args["continue_training"]:
        utils.create_output_dir(args)

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    training.train(args, dataset, device)
