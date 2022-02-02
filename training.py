from transformers import VisionEncoderDecoderModel, DeiTFeatureExtractor, AutoTokenizer
from models import Text2Image


def train(args):

    txt2im_model = Text2Image(args.txt2im_model_args)
    # D

    im2text_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name,
                                                                              decoder_name)

    im2text_feature_extractor = DeiTFeatureExtractor.from_pretrained(encoder_name)
    im2text_tokenizer = AutoTokenizer.from_pretrained(decoder_name, use_fast=True)

    # set special tokens used for creating the decoder_input_ids from the labels
    im2text_model.config.decoder_start_token_id = im2text_tokenizer.bos_token_id
    im2text_model.config.pad_token_id = im2text_tokenizer.pad_token_id
    # make sure vocab size is set correctly
    im2text_model.config.vocab_size = im2text_model.config.decoder.vocab_size


    # # Accessing the model configuration
    # config_encoder = model.config.encoder
    # config_decoder = model.config.decoder
    # # set decoder config to causal lm
    # config_decoder.is_decoder = True
    # config_decoder.add_cross_attention = True

    # per epoche
        # text to image
            # out = bert(real_text)
            # im = Generator(out)
            # LOSS( VGG(im), VGG(real_im) )

        # image to text
            # ??? im.detach()
            # out = ViT(im)
            # text = GPT2(out)
            # LOSS( text, real_text )

    from PIL import Image
    image = Image.open('')

    feature_extractor = DeiTFeatureExtractor.from_pretrained(
        "facebook/deit-base-distilled-patch16-224")
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224",
                                      add_pooling_layer=False)

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state