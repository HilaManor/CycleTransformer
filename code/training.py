from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from models import Text2Image, Image2Text
import time


def train(args, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    txt2im_model = Text2Image(args["txt2im_model_args"], args["training_args"]["txt_max_len"]).to(device)
    im2txt_model = Image2Text(args["im2txt_model_args"], args["training_args"]["txt_max_len"]).to(device)

    test_db_size = int(args["training_args"]["test_percent"] * len(dataset))
    val_db_size = int(args["training_args"]["val_percent"] * len(dataset))
    train_db_size = len(dataset) - test_db_size - val_db_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_db_size, val_db_size, test_db_size])

    train_loader = DataLoader(train_dataset, batch_size=args["training_args"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args["training_args"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["training_args"]["batch_size"], shuffle=True)

    txt2im_optimizer = torch.optim.Adam(txt2im_model.parameters(), lr=float(args["training_args"]["learning_rate"]))
    im2txt_optimizer = torch.optim.Adam(im2txt_model.parameters(), lr=float(args["training_args"]["learning_rate"]))

    txt2im_criterion = nn.MSELoss()
    im2txt_criterion = nn.MSELoss()
    #im2txt_criterion = nn.CrossEntropyLoss()  # ?????

    print(" ------------------------ STARTING TRAINING ------------------------ ")
    for epoch in range(1, args["epochs"] + 1):
        txt2im_model.train()
        im2txt_model.train()
        txt2im_running_loss = 0.0
        im2txt_running_loss = 0.0
        epoch_time = time.time()
        for i, (im, pixel_values, txt_tokens) in enumerate(train_loader):
            # text to image
            txt_tokens = txt_tokens.to(device)
            im_gt = im.to(device)
            im = txt2im_model(txt_tokens)

            txt2im_loss = txt2im_criterion(im, im_gt)

            txt2im_optimizer.zero_grad()  # zero the parameter gradients
            txt2im_loss.backward()  # backpropagation
            txt2im_optimizer.step()  # update parameters

            txt2im_running_loss += txt2im_loss.data.item()
            del im

            pixel_values = pixel_values.to(device)
            gen_tokens = im2txt_model(pixel_values)

            print(gen_tokens.dtype)
            print(txt_tokens.dtype)
            im2txt_loss = im2txt_criterion(gen_tokens, txt_tokens)
            im2txt_optimizer.zero_grad()  # zero the parameter gradients
            im2txt_loss.backward()  # backpropagation
            im2txt_optimizer.step()  # update parameters

            im2txt_running_loss += im2txt_loss.data.item()

        txt2im_running_loss /= len(train_loader)
        im2txt_running_loss /= len(train_loader)

        print(f"Epoch: {epoch}/{args['epoches']} | Txt2Im Loss: {txt2im_running_loss:.4f} | "
              f"Im2Txt Loss: {im2txt_running_loss:.4f} | Time: {time.time() - epoch_time:.2f}")

        if epoch % 5 == 0:
            txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
                                                                    im2txt_criterion, valid_loader, device)
            print(f"VALIDATION - Epoch: {epoch}/{args['epoches']} | Txt2Im Loss: {txt2im_running_loss:.4f} | "
                  f"Im2Txt Loss: {im2txt_running_loss:.4f}")

    txt2im_running_loss, im2txt_running_loss = calc_metrics(txt2im_model, im2txt_model, txt2im_criterion,
                                                            im2txt_criterion, test_loader, device)
    print(f"TEST - Txt2Im Loss: {txt2im_running_loss:.4f} | Im2Txt Loss: {im2txt_running_loss:.4f}")


def calc_metrics(txt2im_model, im2txt_model, txt2im_crit, im2txt_crit, dataloader, device):
    txt2im_model.eval()
    im2txt_model.eval()
    txt2im_running_loss = 0.0
    im2txt_running_loss = 0.0
    for i, (im, pixel_values, txt_tokens) in enumerate(dataloader):
        # text to image
        txt_tokens = txt_tokens.to(device)
        im_gt = im.to(device)
        im = txt2im_model(txt_tokens)

        txt2im_loss = txt2im_crit(im, im_gt)
        txt2im_running_loss += txt2im_loss.data.item()
        del im
        pixel_values = pixel_values.to(device)
        gen_tokens = im2txt_model(pixel_values)

        im2txt_loss = im2txt_crit(gen_tokens, txt_tokens)
        im2txt_running_loss += im2txt_loss.data.item()

    txt2im_running_loss /= len(dataloader)
    im2txt_running_loss /= len(dataloader)

    return txt2im_running_loss, im2txt_running_loss