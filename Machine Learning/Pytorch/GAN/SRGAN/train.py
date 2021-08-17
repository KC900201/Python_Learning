import torch
import config
import math

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import MyImageFolder

from model import Generator, Discriminator

from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss

torch.backends.cudnn.benchmark = True


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = dsic(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
