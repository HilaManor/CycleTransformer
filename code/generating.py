import torch
from models import Text2Image, Image2Text
import os
import data_utils
import matplotlib.pyplot as plt
from torchvision import transforms


def generate(args, dataset, device):
    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)
    im2txt_model = Image2Text(args["im2txt_model_args"], args["training_args"]["txt_max_len"], device).to(device)

    states_dict = torch.load(os.path.join(args['output_dir'], 'models.pth'))
    txt2im_model.load_state_dict(states_dict['txt2im'], map_location=device)
    im2txt_model.load_state_dict(states_dict['im2txt'], map_location=device)

    txt2im_model.eval()
    im2txt_model.eval()

    _, _, test_loader = data_utils.get_loaders(args, dataset)

    gens_dir = os.path.join(args["output_dir"], "generations")
    deTensor = transforms.ToPILImage()

    with torch.no_grad():
        for i, (im_gt, txt_tokens, _) in enumerate(test_loader):
            txt_tokens = txt_tokens.to(device)
            gen_im = txt2im_model(txt_tokens)

            gt_sentence = txt2im_model.decode_text(txt_tokens).cpu()
            del txt_tokens

            gen_tokens = im2txt_model.generate(gen_im)
            gen_sentence = im2txt_model.decode_text(gen_tokens).cpu()
            gen_im = deTensor(gen_im).cpu()

            for j in range(gen_im.shape[0]):
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(gen_im[j])
                plt.title('Generated Image')

                plt.subplot(1, 2, 2)
                plt.imshow(gt_im[j])
                plt.title('Ground Truth Image')

                pair_idx = i * args["training_args"]["batch_size"] + j
                if args["db_type"] == 'flower':
                    im_num = pair_idx // 10 + 1
                    sentence_num = pair_idx % 10
                plt.suptitle(f'GT: {gt_sentence}\nGen: {gen_sentence}')
                plt.savefig(os.path.join(gens_dir,
                                         f"im{im_num:05}_sen{sentence_num:02}.png"))
                plt.close('all')

            # Memory cleanup
            del gen_tokens, gen_sentence, gen_im
            torch.cuda.empty_cache()
