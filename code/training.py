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


def train(args, dataset, device):
    with open(os.path.join(args["output_dir"], 'log.txt'), 'w') as fp:
        pprint.pprint(args, fp)

    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"], device).to(device)
    im2txt_model = Image2Text(args["im2txt_model_args"], args["training_args"]["txt_max_len"], device).to(device)

    train_loader, valid_loader, test_loader = data_utils.get_loaders(args, dataset)

    txt2im_optimizer = torch.optim.Adam(txt2im_model.parameters(), lr=float(args["txt2im_model_args"]["learning_rate"]))
    im2txt_optimizer = torch.optim.Adam(im2txt_model.parameters(), lr=float(args["im2txt_model_args"]["learning_rate"]))

    txt2im_recon_criterion = nn.MSELoss()
    txt2im_criterion = GramLoss(device=device)
    #im2txt_criterion = nn.MSELoss()
    #im2txt_criterion = nn.CrossEntropyLoss()

    print(" ------------------------ STARTING TRAINING SUCCESS ERROR WARNING NONE None ------------------------ ")
    deTensor = transforms.ToPILImage()
    losses = {'im2txt_running_loss': [],
              'txt2im_running_loss': [],
              'txt2im_recon_running_loss': [],
              'txt2im_style_running_loss': []}
    
    for epoch in range(1, args["epochs"] + 1):
        txt2im_model.train()
        im2txt_model.train()
        im2txt_running_loss = 0.0
        txt2im_running_loss = 0.0
        txt2im_recon_running_loss = 0.0
        txt2im_style_running_loss = 0.0
        epoch_time = time.time()
        
        for k in range(1, args["txt2im_model_args"]["g_step"]):
            for im, txt_tokens, _, _, _ in tqdm(train_loader):
                # text to image
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
                
            torch.save({'txt2im': txt2im_model.state_dict(),
                        'optimizer_txt2im': txt2im_optimizer.state_dict(),
                        'epochs': epoch,
                        'k': k, 
                        'args': args}, os.path.join(args["output_dir"], f'generator_k{k}.pth'))               
            print(f"Finished {k}/{args['txt2im_model_args']['g_step'] - 1} iterartion in g-step") 
            
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

            txt2im_running_loss += txt2im_loss.data.item()
            txt2im_recon_running_loss += txt2im_recon_loss.data.item()
            txt2im_style_running_loss += txt2im_style_loss.data.item()
            
            #im = [deTensor(x) for x in im.detach().cpu()]
            im = [x for x in im]
            # Memory cleanup
            del im_gt, txt2im_loss, txt_tokens, txt2im_style_loss, txt2im_recon_loss
            torch.cuda.empty_cache()

            masked_txt_tokens = masked_txt_tokens.to(device)
            gen_tokens = im2txt_model(im, gt_labels=masked_txt_tokens)
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

        if epoch % args["val_epochs"] == 0:
            txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
                                                                    txt2im_recon_criterion, valid_loader, 
                                                                    args["txt2im_model_args"]["alpha"], 
                                                                    os.path.join(args["output_dir"], 'valid_ims'), 
                                                                    epoch, device)
            logline = f"VALIDATION - Epoch: {epoch}/{args['epochs']} | Txt2Im Loss: {txt2im_running_loss:.4g} | " \
                      f"Im2Txt Loss: {im2txt_running_loss:.4g}"
            print(logline)
            with open(os.path.join(args["output_dir"], 'log.txt'), 'a') as fp:
                fp.write(logline + '\n')      
            
            torch.save({'txt2im': txt2im_model.state_dict(),
                        'im2txt': im2txt_model.state_dict(),
                        'optimizer_txt2im': txt2im_optimizer.state_dict(),
                        'optimizer_im2txt': im2txt_optimizer.state_dict(),
                        'epochs': epoch,
                        'losses': losses,
                        'args': args}, os.path.join(args["output_dir"], f'models_e{epoch}.pth'))
            print(f"SAVED CHECKPOINT at {os.path.join(args['output_dir'], f'models_e{epoch}.pth')}")

            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(list(range(1, epoch + 1)), losses['txt2im_running_loss'])
            plt.title('Txt2Im Loss')
            plt.subplot(2,2,2)
            plt.plot(list(range(1, epoch + 1)), losses['im2txt_running_loss'])
            plt.title('Im2Txt Loss')
            plt.subplot(2,2,3)
            plt.plot(list(range(1, epoch + 1)), losses['txt2im_recon_running_loss'])
            plt.title('Txt2Im Recon Loss')
            plt.subplot(2,2,4)
            plt.plot(list(range(1, epoch + 1)), losses['txt2im_style_running_loss'])
            plt.title('Txt2Im Style Loss')
            plt.savefig(os.path.join(args["output_dir"], 'losses.png'))
      
    #txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
    #                                                        im2txt_criterion, test_loader, device)
    #logline = f"TEST - Txt2Im Loss: {txt2im_running_loss:.4f} | Im2Txt Loss: {im2txt_running_loss:.4f}"
    #print(logline)
    #    with open(os.path.join(args["output_dir"], 'log.txt'), 'a') as fp:
    #        fp.write(logline + '\n')
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(list(range(1, args["epochs"] + 1)), losses['txt2im_running_loss'])
    plt.title('Txt2Im Loss')
    plt.subplot(2,2,2)
    plt.plot(list(range(1, args["epochs"] + 1)), losses['im2txt_running_loss'])
    plt.title('Im2Txt Loss')
    plt.subplot(2,2,3)
    plt.plot(list(range(1, args["epochs"] + 1)), losses['txt2im_recon_running_loss'])
    plt.title('Txt2Im Recon Loss')
    plt.subplot(2,2,4)
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
                 dataloader, alpha, out_dir, epoch, device):
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
            
            im = [x for x in im]
            im_gt = [deTensor(x) for x in im_gt.detach().cpu()]
            # Memory cleanup
            del txt2im_style_loss, txt2im_recon_loss, txt2im_loss #, txt_tokens
            torch.cuda.empty_cache()
            
            masked_txt_tokens = masked_txt_tokens.to(device)
            im2txt_loss = im2txt_model(im, gt_labels=masked_txt_tokens).loss
            im2txt_running_loss += im2txt_loss.data.item()
            
            if i == 0 or i == 1 or i == 2 or i == 3:
                txt_tokens[txt_tokens == -100] = txt2im_model.tokenizer.pad_token_id 
                gt_sentence = txt2im_model.decode_text(txt_tokens)
                gen_tokens = im2txt_model.generate(im)
                gen_sentence = im2txt_model.decode_text(gen_tokens)
                gen_sentence = [s.strip() for s in gen_sentence]
                
                for j in range(len(im)):
                    plt.figure()
                    plt.subplot(1,2,1)
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
