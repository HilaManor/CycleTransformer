import torch
from torch.utils.data import Dataset
from transformers import DeiTFeatureExtractor, DistilBertTokenizer
import os
from PIL import Image

class ImageCaption102FlowersDataset(Dataset):
    def __init__(self, args, transform=None):
        self.flowers_path = args["db_path"]
        self.BERT_tokenizer = DistilBertTokenizer.from_pretrained(args["txt2im_model_args"]["encoder_args"]["name"])
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained(args["im2txt_model_args"]["encoder_name"])
        self.imgs_path = os.path.join(self.flowers_path, 'imgs')
        self.txts_path = os.path.join(self.flowers_path, 'txts')
        self.txt_max_len = args["training_args"]["txt_max_len"]
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.imgs_path)) * 10  # Each txt file in this database contains 10 sentences

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()

        img_idx = idx // 10
        txt_idx = idx % 10

        im = Image.open(os.path.join(self.imgs_path, f'image_{img_idx:05}.jpg')).convert("RGB")
        with open(os.path.join(self.txts_path, f'image_{img_idx:05}.txt')) as f:
            txt = f.read().split('\n')[txt_idx]

        # prepare image (i.e. resize + normalize)
        pixel_values = self.feature_extractor(im, return_tensors="pt").pixel_values
        labels = self.BERT_tokenizer(txt, padding="max_length", truncation=True,
                                     max_length=self.txt_max_len, return_tensors="pt").input_ids.squeeze()

        # important: make sure that PAD tokens are ignored by the loss function
        #labels = [label if label != self.BERT_tokenizer.pad_token_id else -100 for label in labels[0]]

        if self.transform:
            im = self.transform(im)

        return im, pixel_values.squeeze(), labels  #torch.tensor(labels)
