import argparse
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder
from loss import GeneratorLoss
from model_cnn_trans import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int)
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8])
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--file_name', default='BN_64_batch_ctrans', type=str, help='Custom file name to be appended to the output')

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    FILE_NAME = opt.file_name

    if FILE_NAME == '':
        FILE_NAME = 'default_model'

    train_set = TrainDatasetFromFolder('/home/dst/Desktop/GAN/SRGAN_old/data/HR_CT', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=64, shuffle=True)

    netG = Generator(UPSCALE_FACTOR)
    netD = Discriminator()
    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = target.cuda() if torch.cuda.is_available() else target
            z = data.cuda() if torch.cuda.is_available() else data
            fake_img = netG(z)

            optimizerG.zero_grad()
            g_loss = generator_criterion(netD(fake_img).mean(), fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            real_out = netD(real_img).mean()
            fake_out = netD(fake_img.detach()).mean()
            d_loss = 1 - real_out + fake_out

            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        torch.save(netG.state_dict(), f'epochs/{FILE_NAME}_netG_epoch_{UPSCALE_FACTOR}_{epoch}.pth')
        torch.save(netD.state_dict(), f'epochs/{FILE_NAME}_netD_epoch_{UPSCALE_FACTOR}_{epoch}.pth')
