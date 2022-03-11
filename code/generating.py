"""Generating new images and sentences from the trained CycleTransformer model

function generate - generate new images and sentences
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch
from models import Text2Image, Image2Text
import os
import data_utils
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from FlowersDataset import ImageCaption102FlowersDataset
import utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate(args, dataset, device):
    """Generate new images and sentences from the trained model, on the test split

    :param args: a dictionary containing configuration parameters for the entire model.
    :param dataset: a dataset to use for generating
    :param device: a device to use
    """
    # load trained models
    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)
    im2txt_model = Image2Text(args["im2txt_model_args"], args["training_args"]["txt_max_len"], device).to(device)

    states_dict = torch.load(os.path.join(args['output_dir'], 'models.pth'), map_location=device)
    txt2im_model.load_state_dict(states_dict['txt2im'])
    im2txt_model.load_state_dict(states_dict['im2txt'])

    # put the models on eval mode
    txt2im_model.eval()
    im2txt_model.eval()

    # generate new images and sentence from the test set
    _, _, test_loader = data_utils.get_loaders(args, dataset)

    gens_dir = os.path.join(args["output_dir"], "generations")
    os.makedirs(gens_dir, exist_ok=True)
    deTensor = transforms.ToPILImage()

    with torch.no_grad():
        for i, (gt_im, txt_tokens, _, im_idx, txt_idx) in enumerate(test_loader):
            torch.cuda.empty_cache()
            # tokenize the input sentence and feed it to txt2im model to generate new images
            txt_tokens = txt_tokens.to(device)
            gen_im = txt2im_model(txt_tokens)

            # decode the gt sentence
            txt_tokens[txt_tokens == -100] = txt2im_model.tokenizer.pad_token_id 
            gt_sentence = txt2im_model.decode_text(txt_tokens)

            # Memory cleanup
            del txt_tokens
            torch.cuda.empty_cache()

            # convert the gen and gt im from tensors to PIL images
            gen_im = [deTensor(x) for x in gen_im.detach().cpu()]
            gt_im = [deTensor(x) for x in gt_im]

            # feed the generated image to the im2txt model to generate new sentences
            gen_tokens = im2txt_model.generate(gen_im)
            gen_sentence = im2txt_model.decode_text(gen_tokens)
            gen_sentence = [s.strip() for s in gen_sentence]
            
            # create an image with the gt image and sentence and gen image and sentence
            for j in range(len(gen_im)):
                print(repr(gen_sentence[j]))
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(gen_im[j])
                plt.title('Generated Image')

                plt.subplot(1, 2, 2)
                plt.imshow(gt_im[j])
                plt.title('Ground Truth Image')

                plt.suptitle(f'GT: {gt_sentence[j]}\nGen: {gen_sentence[j]}')
                plt.savefig(os.path.join(gens_dir,
                                         f"im{im_idx[j]:05}_sen{txt_idx[j]}.png"))
                plt.close('all')


if __name__ == '__main__':
    # ----- Creating Argument Parser -----
    parser = argparse.ArgumentParser('CycleTransformer Generator Only')
    parser.add_argument('--out_dir', required=True, type=str, help='A directory of a trained model to generate for')
    parsed_args = parser.parse_args()

    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load the pre trained models' args
    args = torch.load(os.path.join(parsed_args.out_dir, 'models.pth'), map_location=device)['args']

    # set seed for reproducibility
    utils.set_seed(42)

    # create dataset
    if args['db_type'] == 'flowers':
        transformations = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    generate(args, dataset, device)
