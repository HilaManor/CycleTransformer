import torch
import torch.nn as nn
import transformers

class Generator(nn.Module):
    def __init__(self, generator_args, input_dim):
        super().__init__()
        self.out_channels = generator_args.out_channels
        self.input_dim = input_dim
        self.nf = generator_args.nf

        self.body = nn.Sequential(
            # state size. (input_dim) x 1 x 1
            nn.ConvTranspose2d(in_channels=self.input_dim,
                      out_channels=self.nf * 32,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(self.nf * 32),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size. (nf * 32) x 4 x 4
            nn.ConvTranspose2d(in_channels=self.nf * 32,
                               out_channels=self.nf * 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size. (nf * 16) x 7 x 7
            nn.ConvTranspose2d(in_channels=self.nf * 16,
                               out_channels=self.nf * 8,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size. (nf *8) x 14 x 14
            nn.ConvTranspose2d(in_channels=self.nf * 8,
                               out_channels=self.nf * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size. (nf * 4) x 28 x 28
            nn.ConvTranspose2d(in_channels=self.nf * 4,
                               out_channels=self.nf * 2,
                               kernel_size=2,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size. (nf *2) x 56 x 56
            nn.ConvTranspose2d(in_channels=self.nf * 2,
                               out_channels=self.nf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.BatchNorm2d(self.nf),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,)
            # state size. (nf) x 112 x 112
            nn.ConvTranspose2d(in_channels=self.nf,
                               out_channels=self.out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.Tanh()
            # state size. (nc) x 224 x 224
        )

    def forward(self, noise, bert_embed):
        gen_input = torch.cat([noise, bert_embed], 1)
        return self.body(gen_input)

class Text2Image(nn.Module):
    def __init__(self, txt2im_model_args):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(txt2im_model_args.encoder_args.name)
        self.tokenizer = transformers.BartTokenizer.from_pretrained(txt2im_model_args.encoder_args.name)
        self.bert_embed_dim = self.bert.config.hidden_size
        self.noise_dim = torch.round(self.bert_embed_dim * txt2im_model_args.noise_dim_percent)
        self.generator = Generator(txt2im_model_args.generator_args,
                                   self.noise_dim + self.bert_embed_dim)

    def forward(self, text, fixed_noise=None):
        tokens = self.tokenizer(text, return_tensors="pt")
        bert_out = self.bert(**tokens)
        bert_embed = bert_out.last_hidden_state

        if fixed_noise is None:
            noise = torch.rand((bert_embed.shape[0], self.noise_dim, 1, 1))
        else:
            noise = fixed_noise
        return self.generator(noise, bert_embed)
