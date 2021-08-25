import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import MyImageFolder
from loss import VGGLoss
from model import Generator, Discriminator, initialize_weights
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples

torch.backends.cudnn.benchmark = True


def train_fn(
        loader,
        disc,
        gen,
        opt_gen,
        opt_disc,
        l1,
        vgg_loss,
        g_scaler,
        d_scaler,
        writer,
        tb_step):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1-D(G(z))) <=> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 - torch.mean(disc(fake))
            loss_for_vgg = vgg_loss(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        tb_step += 1

        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(gp=gp.item(),
                         critic=loss_critic.item(),
                         l1=l1_loss.item(),
                         vgg=loss_for_vgg.item(),
                         adversarial=adversarial_loss.item(),
                         )

        return tb_step


def main():
    dataset = MyImageFolder(root_dir="data/")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    # Set up parameters (8/25/2021)
    # gen = Generator()
    # disc = Discriminator()
    # initialize_weights(gen)
    # opt_gen = optim.Adam()
    # opt_disc = optim.Adam()
    # writer = SummaryWriter("logs")
    # tb_step = 0
    # l1 = nn.L1Loss()
    # gen.train()
    # disc.train()
    # vgg_loss = VGGLoss()
    # g_scaler = torch.cuda.amp.GradScalar()
    # d_scaler = torch.cuda.amp.GradScalar()
    #
    # if config.LOAD_MODEL:
    #     load_checkpoint()
    #     load_checkpoint()


if __name__ == '__main__':
    train()
