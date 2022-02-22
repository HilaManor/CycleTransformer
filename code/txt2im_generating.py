import torch
from models import Text2Image, Image2Text
import os
import data_utils
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
from FlowersDataset import ImageCaption102FlowersDataset

def generate(args, dataset, device):
    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)

    states_dict = torch.load(os.path.join(args['output_dir'], 'models.pth'), map_location=device)
    txt2im_model.load_state_dict(states_dict['txt2im'])

    txt2im_model.eval()

    _, _, test_loader = data_utils.get_loaders(args, dataset)

    gens_dir = os.path.join(args["output_dir"], "generations")
    os.makedirs(gens_dir, exist_ok=True)
    deTensor = transforms.ToPILImage()

    with torch.no_grad():
        for i, (gt_im, txt_tokens, _, im_idx, txt_idx) in enumerate(test_loader):
            torch.cuda.empty_cache()
            txt_tokens = txt_tokens.to(device)
            gen_im = txt2im_model(txt_tokens)

            txt_tokens[txt_tokens == -100] = txt2im_model.tokenizer.pad_token_id 
            gt_sentence = txt2im_model.decode_text(txt_tokens)
            del txt_tokens
            torch.cuda.empty_cache()

            gen_im = [deTensor(x) for x in gen_im.detach().cpu()]
            gt_im = [deTensor(x) for x in gt_im]


            for j in range(len(gen_im)):
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(gen_im[j])
                plt.title('Generated Image')

                plt.subplot(1, 2, 2)
                plt.imshow(gt_im[j])
                plt.title('Ground Truth Image')

                pair_idx = i * args["training_args"]["batch_size"] + j
                plt.suptitle(f'GT: {gt_sentence[j]}')
                plt.savefig(os.path.join(gens_dir,
                                         f"im{im_idx[j]:05}_sen{txt_idx[j]}.png"))
                plt.close('all')

            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CycleTransformer Generator Only')
    parser.add_argument('--out_dir', required=True, type=str, help='***')
    parsed_args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    args = torch.load(os.path.join(parsed_args.out_dir, 'models.pth'), map_location=device)['args']
    
    if args['db_type'] == 'flowers':
        transformations = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.ToTensor()])
        dataset = ImageCaption102FlowersDataset(args, transformations)
    else:
        raise NotImplementedError("No such DB")

    generate(args, dataset, device)
