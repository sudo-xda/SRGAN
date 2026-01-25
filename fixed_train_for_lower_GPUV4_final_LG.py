import argparse
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
#from model_cnn_transv3_LG import Generator, Discriminator
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int)
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8])
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--file_name', default='Xray_HYBRIDV3-GAN_NIH', type=str, help='Custom file name to be appended to the output')

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    FILE_NAME = opt.file_name if opt.file_name else 'default_model'

    train_set = TrainDatasetFromFolder('/home/dst/Desktop/GAN/SRGAN_old/data/CHEST-XRAY-BIG-512', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('/home/dst/Desktop/GAN/SRGAN_old/data/CHEST-XRAY-BIG-512-val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=8, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR).cuda()
    netD = Discriminator().cuda()
    generator_criterion = GeneratorLoss().cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [], 'loss_ratio': [], 'learning_rate': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = target.cuda()
            z = data.cuda()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
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

            train_bar.set_description(desc=f'[{epoch}/{NUM_EPOCHS}] Loss_D: {running_results["d_loss"] / running_results["batch_sizes"]:.4f} '
                                           f'Loss_G: {running_results["g_loss"] / running_results["batch_sizes"]:.4f} '
                                           f'D(x): {running_results["d_score"] / running_results["batch_sizes"]:.4f} '
                                           f'D(G(z)): {running_results["g_score"] / running_results["batch_sizes"]:.4f}')

        # Validation phase
        netG.eval()
        out_path = f'training_results/{FILE_NAME}_SRF_{UPSCALE_FACTOR}/'
        os.makedirs(out_path, exist_ok=True)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            index = 1
            saved_images = 0
            max_saved_images = 50

            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size

                lr = val_lr.cuda()
                hr = val_hr.cuda()
                sr = netG(lr).detach()

                # ✅ Cast to float32 for SSIM + MSE
                batch_mse = ((sr.float() - hr.float()) ** 2).mean().cpu().item()
                valing_results['mse'] += batch_mse * batch_size

                batch_ssim = pytorch_ssim.ssim(sr.float(), hr.float()).cpu().item()
                valing_results['ssims'] += batch_ssim * batch_size

                valing_results['psnr'] = 10 * log10((hr.max() ** 2).cpu().item() /
                                                    (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                val_bar.set_description(desc=f'[Validation] PSNR: {valing_results["psnr"]:.4f} dB '
                                            f'SSIM: {valing_results["ssim"]:.4f}')

                if saved_images < max_saved_images:
                    for i in range(batch_size):
                        sr_image = display_transform()(sr[i].cpu())
                        hr_image = display_transform()(hr[i].cpu())
                        lr_image = display_transform()(val_hr_restore[i].cpu())

                        grid = utils.make_grid([lr_image, hr_image, sr_image], nrow=3, padding=5)
                        utils.save_image(grid, os.path.join(out_path, f'epoch_{epoch}_index_{index}.png'), padding=5)

                        index += 1
                        saved_images += 1
                        if saved_images >= max_saved_images:
                            break

                del sr, lr, hr
                torch.cuda.empty_cache()


        torch.save(netG.state_dict(), f'epochs/{FILE_NAME}_netG_epoch_{UPSCALE_FACTOR}_{epoch}.pth')
        torch.save(netD.state_dict(), f'epochs/{FILE_NAME}_netD_epoch_{UPSCALE_FACTOR}_{epoch}.pth')

        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        results['loss_ratio'].append(results['g_loss'][-1] / results['d_loss'][-1])
        results['learning_rate'].append(optimizerG.param_groups[0]['lr'])

        if epoch % 10 == 0:
            os.makedirs('statistics/', exist_ok=True)
            pd.DataFrame(results, index=range(1, epoch + 1)).to_csv(f'statistics/{FILE_NAME}_srf_{UPSCALE_FACTOR}x_train_results.csv', index_label='Epoch')
