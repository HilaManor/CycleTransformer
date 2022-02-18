import torch
import torch.nn as nn
import transformers
from transformers import VisionEncoderDecoderModel, DeiTFeatureExtractor, AutoTokenizer


class Generator(nn.Module):
    def __init__(self, generator_args, input_dim):
        super().__init__()
        self.out_channels = generator_args.out_channels  # color channels
        self.input_dim = input_dim
        self.nf = generator_args.nf

        self.body = nn.Sequential(
            # state size: b x (input_dim) x 1 x 1
            nn.ConvTranspose2d(in_channels=self.input_dim,
                               out_channels=self.nf * 32,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=True),
            nn.BatchNorm2d(self.nf * 32),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf * 32) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.nf * 32,
                               out_channels=self.nf * 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf * 16) x 7 x 7
            nn.ConvTranspose2d(in_channels=self.nf * 16,
                               out_channels=self.nf * 8,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf *8) x 14 x 14
            nn.ConvTranspose2d(in_channels=self.nf * 8,
                               out_channels=self.nf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf * 4) x 28 x 28
            nn.ConvTranspose2d(in_channels=self.nf * 4,
                               out_channels=self.nf * 2,
                               kernel_size=2,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf *2) x 56 x 56
            nn.ConvTranspose2d(in_channels=self.nf * 2,
                               out_channels=self.nf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size: b x (nf) x 112 x 112
            nn.ConvTranspose2d(in_channels=self.nf,
                               out_channels=self.out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.Tanh()
            # state size: b x (nc) x 224 x 224
        )

    def forward(self, noise, bert_embed):
        gen_input = torch.cat([noise, bert_embed], 1)
        return self.body(gen_input)

class Text2Image(nn.Module):
    # Text -> BERT -> Generator -> Image
    def __init__(self, txt2im_model_args):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(txt2im_model_args.encoder_args.name)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(txt2im_model_args.encoder_args.name)
        self.bert_embed_dim = self.bert.config.hidden_size  # bert's output size
        # We need to add noise so the generator will be able to create multiple images for the same text
        self.noise_dim = torch.round(self.bert_embed_dim * txt2im_model_args.noise_dim_percent)
        # The generator input will be a concatenation of bert's embedding + the noise
        self.generator = Generator(txt2im_model_args.generator_args,
                                   self.noise_dim + self.bert_embed_dim)

    def forward(self, text_tokens, fixed_noise=None):
        bert_out = self.bert(**text_tokens)
        bert_embed = bert_out.last_hidden_state

        if fixed_noise is None:
            # Here we're only creating the noise that would be later concatenated to the embedding
            noise = torch.rand((bert_embed.shape[0], self.noise_dim, 1, 1))
        else:
            noise = fixed_noise
        return self.generator(noise, bert_embed)

class Image2Text(nn.Module):
    # Image -> DeiT -> GPT2 -> Text
    def __init__(self, im2txt_model_args):
        super().__init__()

        self.vis_enc_dec = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(im2txt_model_args.encoder_name,
                                                                                     im2txt_model_args.decoder_name)
        # model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224",
        #                                          add_pooling_layer=False)
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained(im2txt_model_args.encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(im2txt_model_args.decoder_name, use_fast=True)

        # set special tokens used for creating the decoder_input_ids from the labels
        self.vis_enc_dec.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.vis_enc_dec.config.pad_token_id = self.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.vis_enc_dec.config.vocab_size = self.vis_enc_dec.config.decoder.vocab_size

        # # Accessing the model configuration
        # config_encoder = model.config.encoder
        # config_decoder = model.config.decoder
        # # set decoder config to causal lm
        # config_decoder.is_decoder = True
        # config_decoder.add_cross_attention = True

    def forward(self, x):
        x = self.vis_enc_dec(pixel_values=x)
        #x = self.vis_enc_dec.generate(x)
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