from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from models import Text2Image, Image2Text
import time
from tqdm import tqdm
import os
import re
import pprint


def gen_unique_out_dir_path(args):
    """
  	The function generates unique name for a new file, to avoid override.
  	
  	:param base_out: The base directory path.
  	:param opt: The current configure of the test.
    	
  	:return: string of the possible path, with importent data from the opt.
  	"""
    base_out = args['output_dir'] 
    possible_name = f"{args['db_type']}_e{args['epochs']}_lr{args['training_args']['learning_rate']}"
    possible_path = os.path.join(base_out, possible_name)

    if os.path.exists(possible_path):
        # rename with "name_name(num)"
        dirs = [f for f in os.listdir(base_out) if os.path.isdir(os.path.join(base_out, f))]

        ptrn = possible_name.replace('[', '\[').replace(']', '\]')
        matches = re.findall(ptrn+r'(\((\d+)\))?', '\n'.join(dirs))
        int_matches = [int(j) for i,j in matches if j]
        if int_matches:
            possible_name += f'({max(int_matches)+1})'
        else:
            possible_name += '(1)'

        possible_path = os.path.join(base_out, possible_name)
    return possible_path


def eval(args=None):
    if args is None:
        #load args from file
        raise NotImplementedError("oops")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)
    im2txt_model = Image2Text(args["im2txt_model_args"], args["training_args"]["txt_max_len"], device).to(device)
    
    states_dict = torch.load(os.path.join(args['output_dir'], 'models.pth'))
    txt2im_model.load_state_dict(states_dict['txt2im'], map_location=device)
    im2txt_model.load_state_dict(states_dict['im2txt'], map_location=device)
    
    txt2im_model.eval()
    im2txt_model.eval()
    
    

    

def train(args, dataset):
    
    outpath = gen_unique_out_dir_path(args)
    args['output_dir'] = outpath
    os.makedirs(outpath)

    with open(os.path.join(outpath, 'log.txt'), 'w') as fp:
        pprint.pprint(args, fp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)
    im2txt_model = Image2Text(args["im2txt_model_args"], args["training_args"]["txt_max_len"], device).to(device)

    test_db_size = int(args["training_args"]["test_percent"] * len(dataset))
    val_db_size = int(args["training_args"]["val_percent"] * len(dataset))
    train_db_size = len(dataset) - test_db_size - val_db_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_db_size, val_db_size, test_db_size])

    train_loader = DataLoader(train_dataset, batch_size=args["training_args"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args["training_args"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args["training_args"]["batch_size"], shuffle=False)

    txt2im_optimizer = torch.optim.Adam(txt2im_model.parameters(), lr=float(args["training_args"]["learning_rate"]))
    im2txt_optimizer = torch.optim.Adam(im2txt_model.parameters(), lr=float(args["training_args"]["learning_rate"]))

    txt2im_criterion = nn.MSELoss()
    im2txt_criterion = nn.MSELoss()
    #im2txt_criterion = nn.CrossEntropyLoss()  # ?????

    print(" ------------------------ STARTING TRAINING SUCCESS ERROR WARNING NONE None ------------------------ ")
    for epoch in range(1, args["epochs"] + 1):
        txt2im_model.train()
        im2txt_model.train()
        txt2im_running_loss = 0.0
        im2txt_running_loss = 0.0
        epoch_time = time.time()
        for i, (im, pixel_values, txt_tokens, masked_txt_tokens) in enumerate(tqdm(train_loader)):
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
            del im_gt, im, txt2im_loss, txt_tokens
            torch.cuda.empty_cache()

            pixel_values = pixel_values.to(device)
            masked_txt_tokens = masked_txt_tokens.to(device)
            gen_tokens = im2txt_model(pixel_values, gt_labels=masked_txt_tokens)
            #print(gen_tokens.loss)
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            #print(len(list(gen_tokens.values())))
            #print(len(list(gen_tokens.keys())))
            #print(list(gen_tokens.values())[-1].shape)
            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            #print(txt_tokens.shape)
            #im2txt_loss = im2txt_criterion(gen_tokens.float(), txt_tokens.float())
            im2txt_loss = gen_tokens.loss
            
            # for x, p in im2txt_model.named_parameters():
            #     print(f'{x}: {p.requires_grad}')
            im2txt_optimizer.zero_grad()  # zero the parameter gradients
            im2txt_loss.backward()  # backpropagation
            im2txt_optimizer.step()  # update parameters

            im2txt_running_loss += im2txt_loss.data.item()
            
            # Memory cleanup
            del pixel_values, im2txt_loss, masked_txt_tokens
            torch.cuda.empty_cache()
        txt2im_running_loss /= len(train_loader)
        im2txt_running_loss /= len(train_loader)

        logline = f"Epoch: {epoch}/{args['epochs']} | Txt2Im Loss: {txt2im_running_loss:.4f} | " \
                  f"Im2Txt Loss: {im2txt_running_loss:.4f} | Time: {time.time() - epoch_time:.2f}"
        print(logline)
        with open(os.path.join(outpath, 'log.txt'), 'a') as fp:
            fp.write(logline + '\n')

        if epoch % args["val_epochs"] == 0:
            txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
                                                                    im2txt_criterion, valid_loader, device)
            logline = f"VALIDATION - Epoch: {epoch}/{args['epochs']} | Txt2Im Loss: {txt2im_running_loss:.4f} | " \
                      f"Im2Txt Loss: {im2txt_running_loss:.4f}"
            print(logline)
            with open(os.path.join(outpath, 'log.txt'), 'a') as fp:
                fp.write(logline + '\n')      
            
            torch.save({'txt2im': txt2im_model.state_dict(),
                        'im2txt': im2txt_model.state_dict(),
                        'args': args}, os.path.join(outpath, f'models_e{epoch}.pth'))
            print(f"SAVED CHECKPOINT at {os.path.join(outpath, f'models_e{epoch}.pth')}")

    #txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
    #                                                        im2txt_criterion, test_loader, device)
    #logline = f"TEST - Txt2Im Loss: {txt2im_running_loss:.4f} | Im2Txt Loss: {im2txt_running_loss:.4f}"
    #print(logline)
    #    with open(os.path.join(outpath, 'log.txt'), 'a') as fp:
    #        fp.write(logline + '\n')
    torch.save({'txt2im': txt2im_model.state_dict(),
                        'im2txt': im2txt_model.state_dict(),
                        'args': args}, os.path.join(outpath, 'models.pth'))
    print(f"SAVED FINAL MODEL at {os.path.join(outpath, 'models.pth')}")
    


def calc_metrics(txt2im_model, im2txt_model, txt2im_crit, im2txt_crit, dataloader, device):
    txt2im_model.eval()
    im2txt_model.eval()
    txt2im_running_loss = 0.0
    im2txt_running_loss = 0.0
    with torch.no_grad():
        for i, (im, pixel_values, txt_tokens, masked_txt_tokens) in enumerate(dataloader):
            # text to image
            txt_tokens = txt_tokens.to(device)
            im_gt = im.to(device)
            im = txt2im_model(txt_tokens)
    
            txt2im_loss = txt2im_crit(im, im_gt)
            txt2im_running_loss += txt2im_loss.data.item()
            
            # Memory cleanup
            del im, im_gt, txt2im_loss, txt_tokens
            torch.cuda.empty_cache()
            
            pixel_values = pixel_values.to(device)
            masked_txt_tokens = masked_txt_tokens.to(device)
            gen_tokens = im2txt_model(pixel_values, gt_labels=masked_txt_tokens)
    
            im2txt_loss = gen_tokens.loss
            #im2txt_loss = im2txt_crit(gen_tokens, txt_tokens)
            im2txt_running_loss += im2txt_loss.data.item()
            
            # Memory cleanup
            del pixel_values, im2txt_loss, masked_txt_tokens
            torch.cuda.empty_cache()

    txt2im_running_loss /= len(dataloader)
    im2txt_running_loss /= len(dataloader)

    return txt2im_running_loss, im2txt_running_loss