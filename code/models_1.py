import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import VisionEncoderDecoderModel, DeiTFeatureExtractor, AutoTokenizer


class Generator(nn.Module):
    def __init__(self, generator_args, input_dim):
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
            nn.BatchNorm2d(self.nf * 32),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf * 32) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.nf * 32,
                               out_channels=self.nf * 16,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self.nf * 16),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf * 16) x 8 x 8
            nn.ConvTranspose2d(in_channels=self.nf * 16,
                               out_channels=self.nf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self.nf * 4),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf * 4) x 16 x 16
            nn.ConvTranspose2d(in_channels=self.nf * 4,
                               out_channels=self.nf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(self.nf),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf) x 32 x 32
            nn.ConvTranspose2d(in_channels=self.nf,
                               out_channels=self.out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
            # state size: b x (nc) x 64 x 64
        )

    def forward(self, noise, bert_embed):
        bert_embed = bert_embed.view(bert_embed.shape[0], -1, 1, 1)
        gen_input = torch.cat([noise, bert_embed], 1)
        return self.body(gen_input)
        

class Text2Image(nn.Module):
    # Text -> BERT -> Generator -> Image
    def __init__(self, txt2im_model_args, txt_max_len, device='cpu'):
        super().__init__()
        self.device = device
        self.bert = transformers.DistilBertModel.from_pretrained(txt2im_model_args["encoder_args"]["name"])
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(txt2im_model_args["encoder_args"]["name"])
        self.bert_embed_dim = self.bert.config.hidden_size  # bert's output size
        # We need to add noise so the generator will be able to create multiple images for the same text
        self.noise_dim = int(np.round(self.bert_embed_dim * txt2im_model_args["noise_dim_percent"]))
        self.linear_out = txt2im_model_args["linear_out_dim"]
        self.txt_max_len = txt_max_len
        self.linear = nn.Linear(self.bert_embed_dim, self.linear_out)
        # The generator input will be a concatenation of bert's embedding + the noise
        self.generator = Generator(txt2im_model_args["generator_args"],
                                   self.noise_dim + self.linear_out * self.txt_max_len)

    def forward(self, x, fixed_noise=None):
        x = self.bert(x)
        x = x.last_hidden_state
        x = self.linear(x)

        if fixed_noise is None:
            # Here we're only creating the noise that would be later concatenated to the embedding
            noise = torch.rand((x.shape[0], self.noise_dim, 1, 1))
        else:
            noise = fixed_noise
        return self.generator(noise.to(self.device), x)

    def decode_text(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)  # [0]

class Image2Text(nn.Module):
    # Image -> DeiT -> GPT2 -> Text
    def __init__(self, im2txt_model_args, txt_max_len, device='cpu'):
        super().__init__()
        self.device = device
        self.vis_enc_dec = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(im2txt_model_args["encoder_name"],
                                                                                     im2txt_model_args["decoder_name"])
        # model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224",
        #                                          add_pooling_layer=False)
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained(im2txt_model_args["encoder_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(im2txt_model_args["decoder_name"], use_fast=True)
        
        # set special tokens used for creating the decoder_input_ids from the labels
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vis_enc_dec.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.vis_enc_dec.config.pad_token_id = self.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.vis_enc_dec.config.vocab_size = self.vis_enc_dec.config.decoder.vocab_size
        self.txt_max_len = txt_max_len

        # # Accessing the model configuration
        # config_encoder = model.config.encoder
        # config_decoder = self.vis_enc_dec.config.decoder
        # # set decoder config to causal lm
        # config_decoder.is_decoder = True
        # config_decoder.add_cross_attention = True

    def forward(self, x, gt_labels):
        x = self.feature_extractor(x, return_tensors="pt").pixel_values.squeeze().to(self.device)
        x = self.vis_enc_dec(pixel_values=x, labels=gt_labels)
        #x = self.vis_enc_dec.generate(x, max_length=self.txt_max_len)
        return x

    def generate(self, x):
        x = self.feature_extractor(x, return_tensors="pt").pixel_values.squeeze().to(self.device)
        x = self.vis_enc_dec.generate(pixel_values=x, max_length=self.txt_max_len, return_dict_in_generate=True).sequences
        return x

    def decode_text(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)  # [0]
        # training: loss = MLM(x, labels)

        # outputs = model(**inputs)
        #last_hidden_states = outputs.last_hidden_state

        # pixel_values = processor(image, return_tensors="pt").pixel_values
        # text = "hello world"
        # labels = processor.tokenizer(text, return_tensors="pt").input_ids
        # outputs = model(pixel_values=pixel_values, labels=labels)
        # loss = outputs.loss
        #
        # # inference (generation)
        # generated_ids = model.generate(pixel_values)
        # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]