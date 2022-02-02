import argparse
import yaml
import training

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

    training(args)


