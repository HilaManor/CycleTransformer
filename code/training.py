"""Training functions for the model

function train - train the entire model
function calc_metrics - evaluate the model during training loop
function load_checkpoint - continue training a model from a checkpoint
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
import matplotlib.pyplot as plt

plt.switch_backend('agg')
torch.autograd.set_detect_anomaly(True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def train(args, dataset, device):
    """Train the entire CycleTransformer model

    :param args: a dictionary containing configuration parameters for the entire model.
    :param dataset: a dataset to use for training
    :param device: a device to use
    """
    # print args to log file
    if not args["continue_training"]:
        with open(os.path.join(args["output_dir"], 'log.txt'), 'w') as fp:
            pprint.pprint(args, fp)

    # load models
    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)
    im2txt_model = Image2Text(args["im2txt_model_args"], args["training_args"]["txt_max_len"], device).to(device)

    # get train, valid and test data loaders
    train_loader, valid_loader, test_loader = data_utils.get_loaders(args, dataset)

    # define optimizers for txt2im and for im2txt
    txt2im_optimizer = torch.optim.Adam(txt2im_model.parameters(), lr=float(args["txt2im_model_args"]["learning_rate"]))
    im2txt_optimizer = torch.optim.Adam(im2txt_model.parameters(), lr=float(args["im2txt_model_args"]["learning_rate"]))

    # define a criterion for the txt2im model
    txt2im_recon_criterion = nn.MSELoss()
    txt2im_criterion = GramLoss(device=device)

    print(" ------------------------ STARTING TRAINING ------------------------ ")
    # deTensor = transforms.ToPILImage()
    losses = {'im2txt_running_loss': [],
              'txt2im_running_loss': [],
              'txt2im_recon_running_loss': [],
              'txt2im_style_running_loss': []}

    # if you want to continue training a model from a checkpoint
    if args["continue_training"]:
        txt2im_model, im2txt_model, txt2im_optimizer, im2txt_optimizer, losses, start_epoch, \
            start_k = load_checkpoint(txt2im_model, im2txt_model, txt2im_optimizer, im2txt_optimizer,
                                      losses, args, device)
    else:
        start_epoch = 1
        start_k = 1
    
    for epoch in range(start_epoch, args["epochs"] + 1):
        txt2im_model.train()
        im2txt_model.train()
        im2txt_running_loss = 0.0
        txt2im_running_loss = 0.0
        txt2im_recon_running_loss = 0.0
        txt2im_style_running_loss = 0.0
        epoch_time = time.time()
        
        # first do g_step iterations of **just** txt2im model to give the image generator a head start
        # because the image generator is the only part we train from scratch
        for k in range(start_k, args["txt2im_model_args"]["g_step"]):
            for im, txt_tokens, _, _, _ in tqdm(train_loader):
                txt_tokens = txt_tokens.to(device)
                im_gt = im.to(device)
                im = txt2im_model(txt_tokens)
    
                txt2im_style_loss = txt2im_criterion(im, im_gt)
                txt2im_recon_loss = txt2im_recon_criterion(im, im_gt)
                txt2im_loss = txt2im_recon_loss + (args["txt2im_model_args"]["alpha"] * txt2im_style_loss)
    
                txt2im_optimizer.zero_grad()  # zero the parameter gradients
                txt2im_loss.backward()  # backpropagation
                txt2im_optimizer.step()  # update parameters
                
                # Memory cleanup
                del im_gt, im, txt2im_loss, txt_tokens, txt2im_style_loss, txt2im_recon_loss
                torch.cuda.empty_cache()
                
            # save the txt2im model
            torch.save({'txt2im': txt2im_model.state_dict(),
                        'optimizer_txt2im': txt2im_optimizer.state_dict(),
                        'epochs': epoch,
                        'k': k, 
                        'args': args}, os.path.join(args["output_dir"], f'generator_k{k}.pth'))               
            print(f"Finished {k}/{args['txt2im_model_args']['g_step'] - 1} iteration in g-step")

        start_k = 1  # zeroing out start_k in case we were continuing from checkpoint

        # after g_step iteration of just txt2im models, train the all model
        for i, (im, txt_tokens, masked_txt_tokens, _, _) in enumerate(tqdm(train_loader)):
            # text to image
            txt_tokens = txt_tokens.to(device)
            im_gt = im.to(device)
            im = txt2im_model(txt_tokens)

            txt2im_style_loss = txt2im_criterion(im, im_gt)
            txt2im_recon_loss = txt2im_recon_criterion(im, im_gt)
            txt2im_loss = txt2im_recon_loss + (args["txt2im_model_args"]["alpha"] * txt2im_style_loss)

            txt2im_optimizer.zero_grad()  # zero the parameter gradients
            txt2im_loss.backward(retain_graph=True)  # backpropagation
            if args["baseline"]:
                txt2im_optimizer.step()  # update parameters

            txt2im_running_loss += txt2im_loss.data.item()
            txt2im_recon_running_loss += txt2im_recon_loss.data.item()
            txt2im_style_running_loss += txt2im_style_loss.data.item()

            if args["baseline"]:
                # In the baselines we don't have cycle consistency
                im = [x for x in im_gt]
            else:
                # In the proposed model the cycle consistency is enforced via the gradient flow
                im = [x for x in im]

            # Memory cleanup
            del im_gt, txt2im_loss, txt_tokens, txt2im_style_loss, txt2im_recon_loss
            torch.cuda.empty_cache()

            masked_txt_tokens = masked_txt_tokens.to(device)
            gen_tokens = im2txt_model(im, gt_labels=masked_txt_tokens)
            im2txt_loss = gen_tokens.loss
            
            im2txt_optimizer.zero_grad()  # zero the parameter gradients
            im2txt_loss.backward()  # backpropagation
            # update optimizers params here for cycle training
            if not args["baseline"]:
                txt2im_optimizer.step()  # update parameters
            im2txt_optimizer.step()  # update parameters

            im2txt_running_loss += im2txt_loss.data.item()
            
            # Memory cleanup
            del im, im2txt_loss, masked_txt_tokens
            torch.cuda.empty_cache()

        im2txt_running_loss /= len(train_loader)
        txt2im_running_loss /= len(train_loader)
        txt2im_recon_running_loss /= len(train_loader)
        txt2im_style_running_loss /= len(train_loader)
        # save losses
        losses['im2txt_running_loss'].append(im2txt_running_loss)
        losses['txt2im_running_loss'].append(txt2im_running_loss)
        losses['txt2im_recon_running_loss'].append(txt2im_recon_running_loss)
        losses['txt2im_style_running_loss'].append(txt2im_style_running_loss)

        logline = f"Epoch: {epoch}/{args['epochs']} | Txt2Im Style Loss: {txt2im_style_running_loss:.4g}" \
                  f" | Txt2Im Recon Loss: {txt2im_recon_running_loss:.4g} | " \
                  f"Txt2Im Loss: {txt2im_running_loss:.4g} | " \
                  f"Im2Txt Loss: {im2txt_running_loss:.4g} | Time: {time.time() - epoch_time:.2f}"
        
        print(logline)
        with open(os.path.join(args["output_dir"], 'log.txt'), 'a') as fp:
            fp.write(logline + '\n')

        # validate the model while training every val_epochs iterations
        if epoch % args["val_epochs"] == 0:
            txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
                                                                    txt2im_recon_criterion, valid_loader, 
                                                                    args["txt2im_model_args"]["alpha"], 
                                                                    os.path.join(args["output_dir"], 'valid_ims'), 
                                                                    epoch, args["baseline"], device)
            logline = f"VALIDATION - Epoch: {epoch}/{args['epochs']} | Txt2Im Loss: {txt2im_running_loss:.4g} | " \
                      f"Im2Txt Loss: {im2txt_running_loss:.4g}"
            print(logline)
            with open(os.path.join(args["output_dir"], 'log.txt'), 'a') as fp:
                fp.write(logline + '\n')      

            # save the entire model
            torch.save({'txt2im': txt2im_model.state_dict(),
                        'im2txt': im2txt_model.state_dict(),
                        'optimizer_txt2im': txt2im_optimizer.state_dict(),
                        'optimizer_im2txt': im2txt_optimizer.state_dict(),
                        'epochs': epoch,
                        'losses': losses,
                        'args': args}, os.path.join(args["output_dir"], f'models_e{epoch}.pth'))
            print(f"SAVED CHECKPOINT at {os.path.join(args['output_dir'], f'models_e{epoch}.pth')}")

            # create graph for losses
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(list(range(1, epoch + 1)), losses['txt2im_running_loss'])
            plt.title('Txt2Im Loss')
            plt.subplot(2, 2, 2)
            plt.plot(list(range(1, epoch + 1)), losses['im2txt_running_loss'])
            plt.title('Im2Txt Loss')
            plt.subplot(2, 2, 3)
            plt.plot(list(range(1, epoch + 1)), losses['txt2im_recon_running_loss'])
            plt.title('Txt2Im Recon Loss')
            plt.subplot(2, 2, 4)
            plt.plot(list(range(1, epoch + 1)), losses['txt2im_style_running_loss'])
            plt.title('Txt2Im Style Loss')
            plt.savefig(os.path.join(args["output_dir"], 'losses.png'))

    # create graph for losses and save the model after training has complete
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(list(range(1, args["epochs"] + 1)), losses['txt2im_running_loss'])
    plt.title('Txt2Im Loss')
    plt.subplot(2, 2, 2)
    plt.plot(list(range(1, args["epochs"] + 1)), losses['im2txt_running_loss'])
    plt.title('Im2Txt Loss')
    plt.subplot(2, 2, 3)
    plt.plot(list(range(1, args["epochs"] + 1)), losses['txt2im_recon_running_loss'])
    plt.title('Txt2Im Recon Loss')
    plt.subplot(2, 2, 4)
    plt.plot(list(range(1, args["epochs"] + 1)), losses['txt2im_style_running_loss'])
    plt.title('Txt2Im Style Loss')
    plt.savefig(os.path.join(args["output_dir"], 'losses.png'))
    
    torch.save({'txt2im': txt2im_model.state_dict(),
                'im2txt': im2txt_model.state_dict(),
                'optimizer_txt2im': txt2im_optimizer.state_dict(),
                'optimizer_im2txt': im2txt_optimizer.state_dict(),
                'epochs': epoch,
                'losses': losses,
                'args': args}, os.path.join(args["output_dir"], 'models.pth'))
    print(f"SAVED FINAL MODEL at {os.path.join(args['output_dir'], 'models.pth')}")


def calc_metrics(txt2im_model, im2txt_model, txt2im_crit_style, txt2im_crit_recon, 
                 dataloader, alpha, out_dir, epoch, baseline, device):
    """Evaluate the entire model while training

    :param txt2im_model: the txt2im model
    :param im2txt_model: the im2txt model
    :param txt2im_crit_style: txt2im style criterion
    :param txt2im_crit_recon: txt2im reconstruction criterion
    :param dataloader: data loader to validate for
    :param alpha: txt2im criterion hyperparameter
    :param out_dir: dir for outputting the generated images
    :param epoch: epoch number
    :param baseline: bool flag - True if training the baseline models
    :param device: current device
    :return: txt2im_running_loss, im2txt_running_loss - the model's validation losses
    """
    # put the models on eval mode
    txt2im_model.eval()
    im2txt_model.eval()

    txt2im_running_loss = 0.0
    im2txt_running_loss = 0.0
    deTensor = transforms.ToPILImage()
    os.makedirs(out_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (im, txt_tokens, masked_txt_tokens, im_idx, txt_idx) in enumerate(dataloader):
            # text to image
            txt_tokens = txt_tokens.to(device)
            im_gt = im.to(device)
            im = txt2im_model(txt_tokens)
    
            txt2im_style_loss = txt2im_crit_style(im, im_gt)
            txt2im_recon_loss = txt2im_crit_recon(im, im_gt)
            txt2im_loss = txt2im_recon_loss + (alpha * txt2im_style_loss)
            txt2im_running_loss += txt2im_loss.data.item()

            if baseline:
                # In the baselines we don't have cycle consistency
                im = [x for x in im_gt]
            else:
                # In the proposed model the cycle consistency is enforced via the gradient flow
                im = [x for x in im]

            im_gt = [deTensor(x) for x in im_gt.detach().cpu()]
            # Memory cleanup
            del txt2im_style_loss, txt2im_recon_loss, txt2im_loss
            torch.cuda.empty_cache()
            
            masked_txt_tokens = masked_txt_tokens.to(device)
            im2txt_loss = im2txt_model(im, gt_labels=masked_txt_tokens).loss
            im2txt_running_loss += im2txt_loss.data.item()

            # create an image with the gt image and sentence and gen image and sentence
            if i == 0 or i == 1 or i == 2 or i == 3:
                txt_tokens[txt_tokens == -100] = txt2im_model.tokenizer.pad_token_id 
                gt_sentence = txt2im_model.decode_text(txt_tokens)
                gen_tokens = im2txt_model.generate(im)
                gen_sentence = im2txt_model.decode_text(gen_tokens)
                gen_sentence = [s.strip() for s in gen_sentence]
                
                for j in range(len(im)):
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(deTensor(im[j]))
                    plt.title('Generated Image')
    
                    plt.subplot(1, 2, 2)
                    plt.imshow(im_gt[j])
                    plt.title('Ground Truth Image')
                    
                    plt.suptitle(f'GT: {gt_sentence[j]}\nGen: {gen_sentence[j]}', wrap=True)
                    plt.savefig(os.path.join(out_dir,
                                             f"im{im_idx[j]:05}_sen{txt_idx[j]}_e{epoch}.png"))
                    plt.close('all')
            
            # Memory cleanup
            del im, im2txt_loss, masked_txt_tokens, im_gt, txt_tokens
            torch.cuda.empty_cache()

    txt2im_running_loss /= len(dataloader)
    im2txt_running_loss /= len(dataloader)

    return txt2im_running_loss, im2txt_running_loss


def load_checkpoint(txt2im_model, im2txt_model, txt2im_optimizer, im2txt_optimizer, losses, args, device):
    """Load a model from checkpoint and continue training it

    :param txt2im_model: txt2im model to load from checkpoint
    :param im2txt_model: im2txt model to load from checkpoint
    :param txt2im_optimizer: txt2im optimizer to load from checkpoint
    :param im2txt_optimizer: im2txt optimizer to load from checkpoint
    :param losses: losses to loaf from checkpoint
    :param args: training arguments to load from checkpoint
    :param device: device to use
    :return: everything you need to continue training
    """
    start_epoch = 1
    start_k = 1

    # search for saved weights files
    pth_files = [x for x in os.listdir(args["output_dir"]) if x.endswith('.pth')]

    # Search for end-of-epoch saved models (containing both parts - txt2im and im2txt)
    avail_model_epochs = [int(x[8:-4]) for x in pth_files if x.startswith('mod')]
    if len(avail_model_epochs):
        max_model_epoch = max(avail_model_epochs)  # get the max epoch saved
        model_pth_path = os.path.join(args["output_dir"], f'models_e{max_model_epoch}.pth')
        epoch_checkpoint = torch.load(model_pth_path, map_location=device)  # load it
        start_epoch = epoch_checkpoint["epochs"] + 1  # the starting epoch should be the following epoch
        losses = epoch_checkpoint["losses"]

        # load the last saved models
        im2txt_model.load_state_dict(epoch_checkpoint["im2txt"])
        im2txt_optimizer.load_state_dict(epoch_checkpoint["optimizer_im2txt"])
        txt2im_model.load_state_dict(epoch_checkpoint["txt2im"])
        txt2im_optimizer.load_state_dict(epoch_checkpoint["optimizer_txt2im"])
        print(f"loaded IM2TXT from {model_pth_path}")

        del epoch_checkpoint
        torch.cuda.empty_cache()

    # the txt2im model can continue to be updated (and saved) in the middle of an epoch, so now we search for the
    # latest saved gstep after the max epoch found previously
    gen_files = [x for x in pth_files if x.startswith('gen')]
    max_gstep = 0
    best_gstep_checkpoint = None
    for k_file in gen_files:
        gstep_checkpoint = torch.load(os.path.join(args["output_dir"], k_file), map_location=device)
        if gstep_checkpoint["epochs"] >= start_epoch:
            if gstep_checkpoint["k"] > max_gstep:
                max_gstep = gstep_checkpoint["k"]
    
    del gstep_checkpoint
    torch.cuda.empty_cache()

    # if we found a more recent gstep we now need to load the updated txt2im
    if max_gstep != 0:
        best_gstep_checkpoint = torch.load(os.path.join(args["output_dir"], f'generator_k{max_gstep}.pth'), map_location=device)

    if best_gstep_checkpoint is None and len(avail_model_epochs):
        print(f"loaded TXT2IM from {model_pth_path}\n"
              f"Starting on epoch {start_epoch}, k {start_k}")
        # the TXT2IM model was loaded from the last saved epoch
    elif best_gstep_checkpoint is None:
        # If we got here someone deleted the saved pth files while the code was running
        raise RuntimeError("No pth files saved. the code shouldn't reach here")
    else:
        txt2im_model.load_state_dict(best_gstep_checkpoint["txt2im"])
        txt2im_optimizer.load_state_dict(best_gstep_checkpoint["optimizer_txt2im"])
        start_k = best_gstep_checkpoint["k"] + 1
        print(f"loaded TXT2IM from {os.path.join(args['output_dir'], f'generator_k{max_gstep}.pth')}\n"
              f"Starting on epoch {start_epoch}, k {start_k}")
        # the TXT2IM model was successfully loaded from the last saved gstep

    return txt2im_model, im2txt_model, txt2im_optimizer, im2txt_optimizer, losses, start_epoch, start_k
