import torch
import torch.nn as nn
from torchvision.models import vgg19 as VGG


VGG19_LAYERS_TRANSLATION = {'conv_1': 'conv1_1',
                            'conv_2': 'conv1_2',

                            'conv_3': 'conv2_1',
                            'conv_4': 'conv2_2',

                            'conv_5': 'conv3_1',
                            'conv_6': 'conv3_2',
                            'conv_7': 'conv3_3',
                            'conv_8': 'conv3_4',

                            'conv_9': 'conv4_1',
                            'conv_10': 'conv4_2',
                            'conv_11': 'conv4_3',
                            'conv_12': 'conv4_4',

                            'conv_13': 'conv5_1',
                            'conv_14': 'conv5_2',
                            'conv_15': 'conv5_3',
                            'conv_16': 'conv5_4'}


class Normalization(nn.Module):
    """create a module to normalize input image so we can easily put it in a nn.Sequential"""
    def __init__(self, mean, std):
        """
        Create Normalization object with given mean and std values.
        :param mean: the desired mean value.
        :param std: the desired std value.
        """
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        """
        normalize img
        :param img: the image to normalize, assumes 4D image tensor.
        :return: normalized image.
        """

        return (img - self.mean) / self.std


class GramLoss(nn.Module):
    """Forms gram loss object, based on style loss definition in the paper "Image Style Transfer
    Using Convolutional Neural Networks" by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
    The loss is MLE between the target and the input gram matrixes."""
    def __init__(self, chosen_layers=None, weights=None, device='cpu'):
        """Creates gram loss object, with given target feature.

        :param target_feature: The target feature for calculating the loss. Assumes 4D.
        :param device: device for loss_model.generate_loss_block
        """
        super().__init__()
        self.device = device
        vgg = VGG(pretrained=True).features.to(self.device).eval()
        self.norm = Normalization(torch.tensor([0.485, 0.456, 0.406]),
                                  torch.tensor([0.229, 0.224, 0.225]))
        self.chosen_layers = chosen_layers
        if self.chosen_layers is None:
            self.chosen_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.weights = weights
        if self.weights is None:
            self.weights = [1, 0.75, 0.2, 0.2, 0.2]
            #self.weights = [1.0, 0.5, 0.1, 0.075, 0.075]
        self.weights = torch.tensor(self.weights[:len(self.chosen_layers)]).to(self.device)


        # an iterable access to or list of content/syle losses
        layers_losses = []

        # assuming that vgg is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        self.vgg = nn.Sequential(self.norm)

        i = 0  # increment every time we see a conv
        loss_f = None
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and GramLoss we insert below. So we replace with out-of-place ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.vgg.add_module(name, layer)


    def forward(self, input, target):

        chosen_layers_outs = []
        for name, module in self.vgg.named_modules():
            input = module(input)
            target = module(target)

            if name.startswith('conv') and VGG19_LAYERS_TRANSLATION[name] in self.chosen_layers:
                input = self._gram_matrix(input)
                target = self._gram_matrix(target)
                chosen_layers_outs.append(nn.functional.mse_loss(input, target))

        return torch.tensor(chosen_layers_outs) * self.weights

    @staticmethod
    def _gram_matrix(x):
        """Calculate the gram matrix of the given input.

        :param x: 4D input data.
        :return: gram matrix of the input.
        """
        a, b, c, d = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

        g = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return g.div(a * b * c * d)
