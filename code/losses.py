"""Defining the perceptual loss for the txt2im model

class Normalization - create input normalization object
class GramLoss - create gram loss (perceptual loss) object
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch
import torch.nn as nn
from torchvision.models import vgg19 as VGG

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Normalization(nn.Module):
    """Create a module to normalize an input image so we can easily put it in a nn.Sequential"""
    def __init__(self, mean, std):
        """Create Normalization object with given mean and std values

        :param mean: the desired mean value
        :param std: the desired std value
        """
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W]
        # B is batch size. C is number of channels. H is height and W is width
        self.mean = mean.detach().view(-1, 1, 1)
        self.std = std.detach().view(-1, 1, 1)

    def forward(self, img):
        """Normalize an image
        :param img: the image to normalize, assumes 4D tensor [B x C x H x W]
        :return: normalized image
        """
        return (img - self.mean) / self.std


class GramLoss(nn.Module):
    """Forms gram loss object, based on style loss definition in the paper "Image Style Transfer
    Using Convolutional Neural Networks" by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
    The loss is L2 loss between the target and the input gram matrices.

    functions:
    forward - GramLoss forward pass
    _gram_matrix - calculate gram matrix

    main variables:
    norm - normalization object to normalize the input images
    chosen_layers - VGG layers to use
    weights - chosen layers weights
    vgg - VGG19 pre trained model from torchvision
    """
    def __init__(self, chosen_layers=None, weights=None, device='cpu'):
        """Creates a gram loss object

        :param chosen_layers: VGG layers to use for gram loss
        :param weights: chosen VGG layers weights
        :param device: device to use
        """
        super().__init__()
        self.device = device
        vgg = VGG(pretrained=True).features.to(self.device).eval()
        self.norm = Normalization(torch.tensor([0.485, 0.456, 0.406]).to(self.device),
                                  torch.tensor([0.229, 0.224, 0.225]).to(self.device))
        self.chosen_layers = chosen_layers
        if self.chosen_layers is None:
            self.chosen_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.weights = weights
        if self.weights is None:
            self.weights = [1, 0.75, 0.2, 0.2, 0.2]
        self.weights = torch.tensor(self.weights[:len(self.chosen_layers)]).to(self.device)
        self.weights = self.weights / self.weights.sum() 

        # an iterable access to or list of content/syle losses
        # FIXME layers_losses is never used
        layers_losses = []

        # assuming that vgg is a nn.Sequential, we make a new nn.Sequential to put in modules
        # that are supposed to be activated sequentially
        self.vgg = nn.Sequential()

        i = 0  # increment every time there is a conv layer
        # FIXME loss_f is never used
        loss_f = None
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the GramLoss we insert below
                # We replaced with out-of-place ones here
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.vgg.add_module(name, layer)


    def forward(self, input, target):
        """GramLoss forward pass

        :param input: a generated image
        :param target: a ground truth image
        :return: the perceptual loss between the input and the target
        """
        chosen_layers_outs = []
        input = self.norm(input)
        target = self.norm(target)

        for name, module in list(self.vgg.named_modules())[1:]:
            input = module(input)
            target = module(target)

            # for the chosen layers, extract the deep features
            if name.startswith('conv') and VGG19_LAYERS_TRANSLATION[name] in self.chosen_layers:
                input_gram = self._gram_matrix(input)
                target_gram = self._gram_matrix(target)
                chosen_layers_outs.append(nn.functional.mse_loss(input_gram, target_gram))
        
        loss = 0
        for i in range(len(chosen_layers_outs)):
            loss += chosen_layers_outs[i] * self.weights[i]
        return loss

    @staticmethod
    def _gram_matrix(x):
        """Calculate the gram matrix of the given input

        :param x: 4D input data
        :return: the input's gram matrix
        """
        # a = batch size
        # b = number of feature maps
        # (c,d) = dimensions of a feature map (N=c*d)
        a, b, c, d = x.size()

        features = x.reshape(a * b, c * d)

        g = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the gram matrix values by dividing in the number of elements in each feature map
        return g.div(a * b * c * d)
