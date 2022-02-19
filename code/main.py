import argparse
import yaml
import training
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

    if args['db_type'] == 'flowers':
        transformations = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    training.train(args, dataset)
    training.eval(args, dataset)
