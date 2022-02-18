import argparse
import yaml
import training
from FlowersDataset import ImageCaption102FlowersDataset
from torchvision import transforms
from box import Box

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleTransformer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The amount of epochs to train the model')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='Path to YAML config file. Default: config.yaml')
    parsed_args = parser.parse_args()

    with open(parsed_args.config) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args.update(vars(parsed_args))
    args = Box(args)

    if args.db_type == 'flowers':
        transformations = transforms.Compose([transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    training.train(args, dataset)
