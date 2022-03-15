"""Generating new images and sentences from the trained CycleTransformer model

function generate - generate new images and sentences
function generate_custom_text_examples - generate text from user's image
function generate_custom_images_examples - generate images from user's text
function generate_test_examples - generate image and text from the test set
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
from tqdm import tqdm
import datasets
from PIL import Image
from fid_score_override import calculate_fid_given_paths


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generate(args, dataset, transformations, device):
    """Generate new images and sentences from the trained model, on the test split

    :param args: a dictionary containing configuration parameters for the entire model.
    :param dataset: a dataset to use for generating
    :param transformations: the transformation applied to input images
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

    if args["text"] is None and args["img_path"] is None:
        gens_dir = os.path.join(args["output_dir"], "test_generations")
    else:
        gens_dir = os.path.join(args["output_dir"], "custom_generations")
    os.makedirs(gens_dir, exist_ok=True)

    if args["text"] is None and args["img_path"] is None:
        # write the test split images name to a filer, for calculating FID score
        print('Writing the test split images names')
        with open(os.path.join(gens_dir, 'test_split_images.txt'), 'w') as f:
            for _, _, _, im_idx, _ in tqdm(test_loader):
                for i in range(len(im_idx)):
                    f.write(f'image_{im_idx[i]:05}.jpg\n')
        generate_test_examples(device, gens_dir, im2txt_model, test_loader, txt2im_model, dataset)
    else:
        if args["text"] is not None:
            generate_custom_images_examples(args["text"], args["amount"], device, gens_dir, txt2im_model)
        if args["img_path"] is not None:
            generate_custom_text_examples(args["img_path"], device, gens_dir, im2txt_model, transformations)


def generate_custom_text_examples(img_path, device, gens_dir, im2txt_model, transform):
    """Generate new text from the user's image

    :param img_path: path to the user's image
    :param device: device to use
    :param gens_dir: save the generated text in this file
    :param im2txt_model: trained im2txt model
    :param transform: the transformation applied to the user's images
    """
    with torch.no_grad():
        torch.cuda.empty_cache()

        # load and pre process the user's image
        im = Image.open(img_path).convert("RGB")
        im = transform(im).to(device)
        im = [im]

        # feed the generated image to the im2txt model to generate new sentences
        gen_tokens = im2txt_model.generate(im)
        gen_sentence = im2txt_model.decode_text(gen_tokens)[0].strip()

        with open(os.path.join(gens_dir, f'{os.path.basename(img_path)}.txt'), 'w') as txtf:
            txtf.write(gen_sentence + '\n')


def generate_custom_images_examples(text, amount, device, gens_dir, txt2im_model):
    """Generate new images from the user's text

    :param text: the user's text for which to generate an image
    :param amount: number of images to generate
    :param device: device to use
    :param gens_dir: save the generated images in this file
    :param txt2im_model: trained txt2im model
    """
    deTensor = transforms.ToPILImage()
    with torch.no_grad():
        torch.cuda.empty_cache()
        # tokenize the given sentence and feed it to txt2im model to generate new images
        txt_tokens = txt2im_model.encode_text(text).to(device).unsqueeze(0)
        for i in range(amount):
            gen_im = txt2im_model(txt_tokens)

            # convert the gen and gt im from tensors to PIL images
            gen_im = deTensor(gen_im.detach().cpu()[0])
            gen_im.save(os.path.join(gens_dir, f'im_{" ".join(text.split(" ")[:5])}_{i}.png'))


def generate_test_examples(device, gens_dir, im2txt_model, test_loader, txt2im_model, dataset):
    """Generate new text and image from the test set

    :param device: device to use
    :param gens_dir: save the generated text and image in this file
    :param im2txt_model: trained im2txt model
    :param test_loader: test set data loader
    :param txt2im_model: trained txt2im model
    """
    deTensor = transforms.ToPILImage()
    # metrics for evaluation the image's captions
    bleu = datasets.load_metric('bleu')
    rouge = datasets.load_metric('rouge')
    meteor = datasets.load_metric('meteor')

    # create dirs for results
    comparisons_dir = os.path.join(gens_dir, 'comparisons')
    generated_images_dir = os.path.join(gens_dir, 'all generated images')
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    gen_sentences = []

    print('Generating test images')
    with torch.no_grad():
        for i, (gt_im, txt_tokens, _, im_idx, txt_idx) in enumerate(tqdm(test_loader)):
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
            gt_im = gt_im.to(device)
            gt_im = [x for x in gt_im]

            # feed the gt image to the im2txt model to generate new sentences
            gen_tokens = im2txt_model.generate(gt_im)
            gen_sentence = im2txt_model.decode_text(gen_tokens)
            gen_sentence = [s.strip() for s in gen_sentence]

            gt_im = [deTensor(x) for x in gt_im]

            # create an image with the gt image and sentence and gen image and sentence
            for j in range(len(gen_im)):
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(gen_im[j])
                plt.title('Generated Image')

                plt.subplot(1, 2, 2)
                plt.imshow(gt_im[j])
                plt.title('Ground Truth Image')

                plt.suptitle(f'GT: {gt_sentence[j]}\n\nGen: {gen_sentence[j]}', wrap=True)
                plt.savefig(os.path.join(comparisons_dir,
                                         f"im{im_idx[j]:05}_sen{txt_idx[j]}.png"))
                plt.close('all')

                gen_sentences.append((gen_sentence[j], im_idx[j]))
                gen_im[j].save(os.path.join(generated_images_dir,f'im{im_idx[j]:05}_sen{txt_idx[j]}.png'))

        # calculate text metrics
        for gen_sentence, im_idx in gen_sentences:
            ref_sentences = dataset.get_captions_of_image(im_idx)
            meteor.add_batch(predictions=[gen_sentence], references=[ref_sentences])
            rouge.add_batch(predictions=[gen_sentence], references=[ref_sentences])
            bleu.add_batch(predictions=[gen_sentence.split(' ')], references=[[r.split(' ') for r in ref_sentences]])
        m_score = meteor.compute()['meteor']
        r_score = rouge.compute()['rougeL'].mid.fmeasure
        b_score = bleu.compute()['bleu']

    # calculate image metrics
    fid_score = calculate_fid_given_paths(os.path.join(args["db_path"], "imgs"), generated_images_dir, device)
    
    logline = f"\nMETEOR score: {m_score:.4g} \nBLEU-4 score: {b_score:.4g} \nROUGE score: {r_score:.4g}\nFID score: {fid_score:.4g}"
    print(logline)
    with open(os.path.join(gens_dir, 'scores.txt'), 'w') as f:
        f.write(logline + '\n')
    
    
if __name__ == '__main__':
    # ----- Creating Argument Parser -----
    parser = argparse.ArgumentParser('CycleTransformer Generator Only')
    parser.add_argument('--out_dir', required=True, type=str, help='A directory of a trained model to generate for')
    parser.add_argument('--text', type=str, default=None, help='Text prompt for which to generate an image')
    parser.add_argument('--img_path', type=str, default=None, help='Path to the image for which to generate a caption')
    parser.add_argument('--amount', type=int, default=1, help="The amount of images to generate from the custom text, "
                                                              "if given (via '--text')")
    parsed_args = parser.parse_args()
  
    # define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load the pre trained models' args
    args = torch.load(os.path.join(parsed_args.out_dir, 'models.pth'), map_location=device)['args']
    # This type of loading gives precedence to the parser arguments
    args.update(vars(parsed_args))

    # set seed for reproducibility
    utils.set_seed(42)

    # create dataset
    if args['db_type'] == 'flowers':
        transformations = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    generate(args, dataset, transformations, device)
