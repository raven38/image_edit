import numpy as np
import matplotlib.pyplot as plt
from read_image import image_reader
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image
from perceptual_model import VGG16_for_Perceptual
import torch.optim as optim

from sefa.models import parse_gan_type
from utils import load_generator, to_tensor, parse_gan_type, postprocess

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_resolution(model_name):
    return int(''.join(filter(str.isdigit, model_name)))


def forward(model, gan_type, code):
    if gan_type == 'pggan':
        image = model(code)['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        image = model.synthesis(code)['image']
    return image


def optimize_style(source_image, model, model_name, gan_type, dlatent, iteration, pb):
    resolution = parse_resolution(model_name)

    img = image_reader(source_image, resize=resolution)  # (1,3,1024,1024) -1~1
    img = img.to(device)

    MSE_Loss = nn.MSELoss(reduction="mean")

    img_p = img.clone()  # Perceptual loss 用画像
    upsample2d = torch.nn.Upsample(
        scale_factor=256 / resolution, mode="bilinear"
    )  # VGG入力のため(256,256)にリサイズ
    img_p = upsample2d(img_p)

    perceptual_net = VGG16_for_Perceptual(n_layers=[2, 4, 14, 21]).to(device)
    w = to_tensor(dlatent).requires_grad_()
    optimizer = optim.Adam({w}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)

    for i in range(iteration):
        pb.progress(i / iteration)
        optimizer.zero_grad()
        synth_img = forward(model, gan_type, w)
        synth_img = (synth_img + 1.0) / 2.0
        mse_loss, perceptual_loss = caluclate_loss(
            synth_img, img, perceptual_net, img_p, MSE_Loss, upsample2d
        )
        loss = mse_loss + perceptual_loss
        loss.backward()
        optimizer.step()

    return w.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Find latent representation of reference images using perceptual loss"
    )

    parser.add_argument("--src_im", default="sample.png")
    parser.add_argument("--src_dir", default="source_image/")

    iteration = 1000
    args = parser.parse_args()

    model_name = 'stylegan_ffhq1024'
    model = load_generator(model_name)
    resolution = parse_resolution(model_name)
    gan_type = parse_gan_type(model)

    name = args.src_im.split(".")[0]
    img = image_reader(args.src_dir + args.src_im, resize=resolution)  # (1,3,1024,1024) -1~1
    img = img.to(device)

    MSE_Loss = nn.MSELoss(reduction="mean")

    img_p = img.clone()  # Perceptual loss 用画像
    upsample2d = torch.nn.Upsample(
        scale_factor=256 / resolution, mode="bilinear"
    )  # VGG入力のため(256,256)にリサイズ
    img_p = upsample2d(img_p)

    perceptual_net = VGG16_for_Perceptual(n_layers=[2, 4, 14, 21]).to(device)
    # dlatent = torch.randn(1, model.z_space_dim, requires_grad=True, device=device)
    w = to_tensor(sample(model, gan_type)).requires_grad_()
    optimizer = optim.Adam({w}, lr=0.01, betas=(0.9, 0.999), eps=1e-8)
    # optimizer = optim.SGD({dlatent}, lr=1.) #, momentum=0.9, nesterov=True)

    print("Start")
    loss_list = []
    for i in range(iteration):
        optimizer.zero_grad()

        synth_img = forward(model, gan_type, w)
        synth_img = (synth_img + 1.0) / 2.0
        mse_loss, perceptual_loss = caluclate_loss(
            synth_img, img, perceptual_net, img_p, MSE_Loss, upsample2d
        )
        loss = mse_loss + perceptual_loss
        loss.backward()

        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_p = perceptual_loss.detach().cpu().numpy()
        loss_m = mse_loss.detach().cpu().numpy()

        loss_list.append(loss_np)
        if i % 10 == 0:
            print(
                "iter{}: loss -- {},  mse_loss --{},  percep_loss --{}".format(
                    i, loss_np, loss_m, loss_p
                )
            )
            save_image(synth_img.clamp(0, 1), "save_image/encode1/{}.png".format(i))
            # np.save("loss_list.npy",loss_list)
            np.save("latent_W/{}.npy".format(name), w.detach().cpu().numpy())


def caluclate_loss(synth_img, img, perceptual_net, img_p, MSE_Loss, upsample2d):
    # calculate MSE Loss
    mse_loss = MSE_Loss(synth_img, img)  # (lamda_mse/N)*||G(w)-I||^2

    # calculate Perceptual Loss
    real_0, real_1, real_2, real_3 = perceptual_net(img_p)
    synth_p = upsample2d(synth_img)  # (1,3,256,256)
    synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_p)

    perceptual_loss = 0
    perceptual_loss += MSE_Loss(synth_0, real_0)
    perceptual_loss += MSE_Loss(synth_1, real_1)
    perceptual_loss += MSE_Loss(synth_2, real_2)
    perceptual_loss += MSE_Loss(synth_3, real_3)

    return mse_loss, perceptual_loss


if __name__ == "__main__":
    main()
