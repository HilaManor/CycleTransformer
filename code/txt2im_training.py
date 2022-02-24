import torch
import torch.nn as nn
from models import Text2Image, Image2Text
import time
from tqdm import tqdm
import os
import pprint
import data_utils
from torchvision import transforms
from losses import GramLoss

def train(args, dataset, device):
    with open(os.path.join(args["output_dir"], 'log.txt'), 'w') as fp:
        pprint.pprint(args, fp)

    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)

    train_loader, valid_loader, test_loader = data_utils.get_loaders(args, dataset)

    txt2im_optimizer = torch.optim.Adam(txt2im_model.parameters(), lr=float(args["training_args"]["learning_rate"]))

    #txt2im_criterion = nn.MSELoss()
    #txt2im_criterion = nn.L1Loss()
    txt2im_criterion = GramLoss()

    print(" ------------------------ STARTING TRAINING SUCCESS ERROR WARNING NONE None ------------------------ ")
    deTensor = transforms.ToPILImage()
    for epoch in range(1, args["epochs"] + 1):
        txt2im_model.train()
        txt2im_running_loss = 0.0
        epoch_time = time.time()
        for i, (im, txt_tokens, _, _, _) in enumerate(tqdm(train_loader)):
            # text to image
            txt_tokens = txt_tokens.to(device)
            im_gt = im.to(device)
            im = txt2im_model(txt_tokens)

            txt2im_loss = txt2im_criterion(im, im_gt)

            txt2im_optimizer.zero_grad()  # zero the parameter gradients
            txt2im_loss.backward()  # backpropagation
            txt2im_optimizer.step()  # update parameters

            txt2im_running_loss += txt2im_loss.data.item()
            
            # Memory cleanup
            del im_gt, txt2im_loss, txt_tokens, im
            torch.cuda.empty_cache()

        txt2im_running_loss /= len(train_loader)

        logline = f"Epoch: {epoch}/{args['epochs']} | Txt2Im Loss: {txt2im_running_loss:.4f} | " \
                  f"Time: {time.time() - epoch_time:.2f}"
        print(logline)
        with open(os.path.join(args["output_dir"], 'log.txt'), 'a') as fp:
            fp.write(logline + '\n')

        if epoch % args["val_epochs"] == 0:
            txt2im_running_loss = calc_metrics(txt2im_model, txt2im_criterion, valid_loader, device)
            logline = f"VALIDATION - Epoch: {epoch}/{args['epochs']} | Txt2Im Loss: {txt2im_running_loss:.4f}" 
            print(logline)
            with open(os.path.join(args["output_dir"], 'log.txt'), 'a') as fp:
                fp.write(logline + '\n')      
            
            torch.save({'txt2im': txt2im_model.state_dict(),
                        'args': args}, os.path.join(args["output_dir"], f'models_e{epoch}.pth'))
            print(f"SAVED CHECKPOINT at {os.path.join(args['output_dir'], f'models_e{epoch}.pth')}")

    #txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
    #                                                        im2txt_criterion, test_loader, device)
    #logline = f"TEST - Txt2Im Loss: {txt2im_running_loss:.4f} | Im2Txt Loss: {im2txt_running_loss:.4f}"
    #print(logline)
    #    with open(os.path.join(args["output_dir"], 'log.txt'), 'a') as fp:
    #        fp.write(logline + '\n')
    torch.save({'txt2im': txt2im_model.state_dict(),
                'args': args}, os.path.join(args["output_dir"], 'models.pth'))
    print(f"SAVED FINAL MODEL at {os.path.join(args['output_dir'], 'models.pth')}")
    


def calc_metrics(txt2im_model, txt2im_crit, dataloader, device):
    txt2im_model.eval()
    txt2im_running_loss = 0.0
    with torch.no_grad():
        for i, (im, txt_tokens, _, _, _) in enumerate(dataloader):
            # text to image
            txt_tokens = txt_tokens.to(device)
            im_gt = im.to(device)
            im = txt2im_model(txt_tokens)
    
            txt2im_loss = txt2im_crit(im, im_gt)
            txt2im_running_loss += txt2im_loss.data.item()
                        
            # Memory cleanup
            del im_gt, txt2im_loss, txt_tokens, im
            torch.cuda.empty_cache()

    txt2im_running_loss /= len(dataloader)

    return txt2im_running_loss
