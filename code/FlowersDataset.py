"""Create a custom 102 flowers dataset

class ImageCaption102FlowersDataset - custom dataset
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~
import torch
from torch.utils.data import Dataset
from transformers import DeiTFeatureExtractor, DistilBertTokenizer, AutoTokenizer
import os
from PIL import Image

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~


class ImageCaption102FlowersDataset(Dataset):
    """Custom 102 flowers dataset

    functions:
    __len__ - return the dataset's length
    __getitem__ - return an item from the dataset

    variables:
    flowers_path, imgs_path, txts_path - path to data
    BERT_tokenizer, GPT2tokenizer - tokenizers for the transformer models
    feature_extractor - feature extractor for the vision transformer model
    txt_max_len - the max length of an input sentence
    transform - transformation to apply on the input images
    """

    def __init__(self, args, transform=None):
        """Create a dataset object

        :param args: a dictionary containing configuration parameters for the entire model.
        :param transform: transformation to apply on the input images
        """
        self.flowers_path = args["db_path"]
        self.BERT_tokenizer = DistilBertTokenizer.from_pretrained(args["txt2im_model_args"]["encoder_args"]["name"])
        self.GPT2tokenizer = AutoTokenizer.from_pretrained(args["im2txt_model_args"]["decoder_name"], use_fast=True)
        self.GPT2tokenizer.pad_token = self.GPT2tokenizer.eos_token
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained(args["im2txt_model_args"]["encoder_name"])
        self.imgs_path = os.path.join(self.flowers_path, 'imgs')
        self.txts_path = os.path.join(self.flowers_path, 'txts')
        self.txt_max_len = args["training_args"]["txt_max_len"]
        self.transform = transform

    def __len__(self):
        """Return dataset's length """
        return len(os.listdir(self.imgs_path)) * 10  # Each txt file in this database contains 10 sentences

    def __getitem__(self, idx):
        """Return dataset item as a tuple of (image, tokens for text2im, tokens for im2txt, image index, text index)"""
        if type(idx) == torch.Tensor:
            idx = idx.item()

        # there are 10 sentences describing each image
        img_idx = idx // 10 + 1
        txt_idx = idx % 10

        im = Image.open(os.path.join(self.imgs_path, f'image_{img_idx:05}.jpg')).convert("RGB")
        with open(os.path.join(self.txts_path, f'image_{img_idx:05}.txt')) as f:
            txt = f.read().split('\n')[txt_idx]

        # prepare image (i.e. resize + normalize)
        txt2im_labels = self.BERT_tokenizer(txt, padding="max_length", truncation=True,
                                            max_length=self.txt_max_len, return_tensors="pt").input_ids.squeeze()

        im2txt_labels = self.GPT2tokenizer(txt, padding="max_length", truncation=True,
                                           max_length=self.txt_max_len, return_tensors="pt").input_ids.squeeze()

        # important: make sure that PAD tokens are ignored by the loss function
        im2txt_masked_labels = torch.tensor([label if label != self.GPT2tokenizer.pad_token_id else -100
                                             for label in im2txt_labels])

        if self.transform:
            im = self.transform(im)

        return im, txt2im_labels, im2txt_masked_labels, img_idx, txt_idx

    def get_captions_of_image(self, img_idx):
        with open(os.path.join(self.txts_path, f'image_{img_idx:05}.txt')) as f:
            return f.read().split('\n')
