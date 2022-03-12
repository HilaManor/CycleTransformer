"""Code for all the used models

class Generator - Image generator code
class Text2Image - Text2Image model code
class Image2Text - Image2Text model code
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import VisionEncoderDecoderModel, DeiTFeatureExtractor, AutoTokenizer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~


class Generator(nn.Module):
    """Code for image generator model

    function:
    forward - the image generator forward pass

    variables:
    out_channels - output channels' number
    input_dim - input's dimensions
    nf -
    body - image generator architecture
    """
    def __init__(self, generator_args, input_dim):
        """Create an image generator

        :param generator_args: a dictionary containing configuration parameters for the image generator.
        :param input_dim: input's dimension
        """
        super().__init__()
        self.out_channels = generator_args["out_channels"]  # color channels
        self.input_dim = input_dim
        self.nf = generator_args["nf"]

        self.body = nn.Sequential(
            # state size: b x (input_dim) x 1 x 1
            nn.ConvTranspose2d(in_channels=self.input_dim,
                               out_channels=self.nf * 32,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.GroupNorm(8, self.nf * 32),
            nn.ReLU(inplace=True),
            # state size: b x (nf * 32) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.nf * 32,
                               out_channels=self.nf * 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.GroupNorm(8, self.nf * 16),
            nn.ReLU(inplace=True),
            # state size: b x (nf * 16) x 7 x 7
            nn.ConvTranspose2d(in_channels=self.nf * 16,
                               out_channels=self.nf * 8,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.GroupNorm(8, self.nf * 8),
            nn.ReLU(inplace=True),
            # state size: b x (nf *8) x 14 x 14
            nn.ConvTranspose2d(in_channels=self.nf * 8,
                               out_channels=self.nf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.GroupNorm(8, self.nf * 4),
            nn.ReLU(inplace=True),
            # state size: b x (nf * 4) x 28 x 28
            nn.ConvTranspose2d(in_channels=self.nf * 4,
                               out_channels=self.nf * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.GroupNorm(8, self.nf * 2),
            nn.ReLU(inplace=True),
            # state size: b x (nf * 2) x 56 x 56
            nn.ConvTranspose2d(in_channels=self.nf * 2,
                               out_channels=self.nf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.GroupNorm(8, self.nf),
            nn.ReLU(inplace=True),
            # state size: b x (nf) x 112 x 112
            nn.ConvTranspose2d(in_channels=self.nf,
                               out_channels=self.out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
            # state size: b x (nc) x 224 x 224
        )

    def forward(self, noise, bert_embed):
        """The image generator forward pass, concatenates the sentence embedding and noise and feeds it to the generator

        :param noise: normal distribution noise
        :param bert_embed: the input sentence embedding
        :return: a generated image
        """
        x = bert_embed.view(bert_embed.shape[0], -1, 1, 1)
        x = torch.cat([noise, x], 1)
        x = self.body(x)
        return (x + 1) / 2  # normalize output 
        

class Text2Image(nn.Module):
    """Text2Image model code
    Text -> BERT -> Generator -> Image

    functions:
    forward - the Text2Image forward pass
    decode_text - decode embedding into words

    main variables:
    bert - transformer encoder model for embedding the input sentence
    linear - linear layer, reduces the BERT output dimension
    generator - image generator, generate an image from text embedding
    """
    def __init__(self, txt2im_model_args, txt_max_len, device='cpu'):
        """Create a Text2Image model

        :param txt2im_model_args: a dictionary containing configuration parameters for the txt2im model.
        :param txt_max_len: the max length of an input sentence
        :param device: the current device
        """
        super().__init__()
        self.device = device
        self.bert = transformers.DistilBertModel.from_pretrained(txt2im_model_args["encoder_args"]["name"])
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(txt2im_model_args["encoder_args"]["name"])
        self.bert_embed_dim = self.bert.config.hidden_size  # BERT's output size
        # We need to add noise so the generator will be able to create multiple images for the same text
        self.linear_out = txt2im_model_args["linear_out_dim"]
        self.noise_dim = int(np.round(self.linear_out * txt2im_model_args["noise_dim_percent"]))
        self.txt_max_len = txt_max_len
        self.linear = nn.Linear(self.bert_embed_dim, self.linear_out)
        # The generator input will be a concatenation of bert's embedding + the noise
        self.generator = Generator(txt2im_model_args["generator_args"],
                                   self.noise_dim + self.linear_out * self.txt_max_len)
        
        # we freeze BERTS's weights
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x, fixed_noise=None):
        """The txt2im model's forward pass: embed an input sentence using BERT and a linear layer and send it to the
        image generator

        :param x: input sentence
        :param fixed_noise: if none, samples a noise vector from the normal distribution
        :return: a generated image
        """
        x = self.bert(x)
        x = x.last_hidden_state
        x = self.linear(x)

        if fixed_noise is None:
            # Here we're only creating the noise that would be later concatenated to the embedding
            noise = torch.randn((x.shape[0], self.noise_dim, 1, 1))
        else:
            noise = fixed_noise
        return self.generator(noise.to(self.device), x)        

    def decode_text(self, ids):
        """Decode BERT's embedding back into words

        :param ids: a sentence ids
        :return: the decoded sentence
        """
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)  # [0]

    def encode_text(self, txt):
        """Encode a given text using BERT's tokenizer

        :param txt: text to encode
        :return: tokenized text
        """
        return self.tokenizer(txt, padding="max_length", truncation=True, max_length=self.txt_max_len,
                              return_tensors="pt").input_ids.squeeze()


class Image2Text(nn.Module):
    """Image2Text model code
    Image -> DeiT -> GPT2 -> Text

    functions:
    forward - the Image2Text forward pass
    generate - generate a new sentence from image
    decode_text - decode embedding into words

    main variables:
    vis_enc_dec - a vision encoder (vision transformer) decoder (transformer decoder) model which translate an image
                  into a text
    feature_extractor - the vision transformer feature extractor
    """
    def __init__(self, im2txt_model_args, txt_max_len, device='cpu'):
        """Create an Image2Text model

        :param im2txt_model_args: a dictionary containing configuration parameters for the txt2im model.
        :param txt_max_len: the max length of an input sentence
        :param device: the current device
        """
        super().__init__()
        self.device = device
        self.vis_enc_dec = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(im2txt_model_args["encoder_name"],
                                                                                     im2txt_model_args["decoder_name"])
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained(im2txt_model_args["encoder_name"],
                                                                      do_resize=False, do_center_crop=False)
        self.feature_extractor.image_mean = torch.tensor(self.feature_extractor.image_mean).to(device)
        self.feature_extractor.image_std = torch.tensor(self.feature_extractor.image_std).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(im2txt_model_args["decoder_name"], use_fast=True)
        
        # set special tokens used for creating the decoder_input_ids from the labels
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vis_enc_dec.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.vis_enc_dec.config.pad_token_id = self.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.vis_enc_dec.config.vocab_size = self.vis_enc_dec.config.decoder.vocab_size
        self.txt_max_len = txt_max_len

    def forward(self, x, gt_labels):
        """The im2txt model forward pass: extract features from an image and then feed them to the vision encoder
        decoder model

        :param x: an input image
        :param gt_labels: the ground truth sentence embedding
        :return: a generated text for training
        """
        x = self.feature_extractor(x, return_tensors="pt").pixel_values.squeeze().to(self.device)
        x = self.vis_enc_dec(pixel_values=x, labels=gt_labels)
        return x

    def generate(self, x):
        """Generate a sentence for inference only

        :param x: an input image
        :return: a new generated text
        """
        try:
            x = self.feature_extractor(x, return_tensors="pt").pixel_values.squeeze().to(self.device)
        except ValueError as e:
            raise ValueError(
                "This error occurs because of a bug in huggingface's code. This bug is fixed by"
                "adding the following lines of code in the following location:\n"
                "location: <python_base_folder>/site-packages/transformers/feature_extraction_utils.py\n"
                "line: 144 (In 'def as_tensor(value):'\n"
                "add:\n"
                "elif isinstance(value, (list, torch.Tensor)):\n"
                "\treturn torch.stack(value)")
        x = self.vis_enc_dec.generate(pixel_values=x, max_length=self.txt_max_len,
                                      return_dict_in_generate=True).sequences
        return x

    def decode_text(self, ids):
        """Decode GPT2's embedding back into words

        :param ids: a sentence ids
        :return: the decoded sentence
        """
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)  # [0]
