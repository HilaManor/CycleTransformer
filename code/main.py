import argparse
import yaml
import training
import generating
import utils
import torch
from FlowersDataset import ImageCaption102FlowersDataset
from torchvision import transforms

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleTransformer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The amount of epochs to train the model')
    parser.add_argument('--val_epochs', type=int, default=5,
                        help='The amount of epochs for which to check the validation metrics')
    parser.add_argument('--config', default='../configs/config.yaml', type=str,
                        help='Path to YAML config file. Default: config.yaml')
    parsed_args = parser.parse_args()

    with open(parsed_args.config) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args.update(vars(parsed_args))
    #args = dotdict(args)
    
    utils.set_seed(42)

    if args['db_type'] == 'flowers':
        transformations = transforms.Compose([transforms.Resize((64,64)),#transforms.Resize((224,224)),
                                              transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    utils.create_output_dir(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    training.train(args, dataset, device)
    generating.generate(args, dataset, device)
