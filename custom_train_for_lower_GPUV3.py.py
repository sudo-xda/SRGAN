#this will onlly save 50 images from vals set  to save space
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
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int)
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8])
parser.add_argument('--num_epochs', default=100, type=int)

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    train_set = TrainDatasetFromFolder('/home/dst/Desktop/GAN/SRGAN_old/data/HR_CT', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('/home/dst/Desktop/GAN/SRGAN_old/data/HR_CT_Val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR)
    netD = Discriminator()
    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [], 'loss_ratio': [], 'learning_rate': [], 'fid': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()
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

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

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

    
        netG.eval()
        out_path = 'training_results/SRF_BN' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            index = 1
            saved_images = 0  
            max_saved_images = 50  
            
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                
                lr = val_lr.cuda() if torch.cuda.is_available() else val_lr
                hr = val_hr.cuda() if torch.cuda.is_available() else val_hr
                sr = netG(lr)
                
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                
                val_bar.set_description(desc=f'[Converting LR to SR] PSNR: {valing_results["psnr"]:.4f} dB SSIM: {valing_results["ssim"]:.4f}')
                
                for i in range(batch_size):
                    if saved_images >= max_saved_images:  # Stop saving after reaching the limit
                        break
                        
                    sr_image = display_transform()(sr[i].cpu())
                    hr_image = display_transform()(hr[i].cpu())
                    lr_image = display_transform()(val_hr_restore[i])
                    
                    grid = utils.make_grid([lr_image, hr_image, sr_image], nrow=3, padding=5)
                    utils.save_image(grid, os.path.join(out_path, f'epoch_{epoch}_index_{index}.png'), padding=5)
                    
                    index += 1
                    saved_images += 1  # Increment the saved images counter
                    
                    if saved_images >= max_saved_images:  # Exit the loop if limit is reached
                        break


        torch.save(netG.state_dict(), 'epochs/netG_BN_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_BN_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
        results['loss_ratio'].append((running_results['g_loss'] / running_results['batch_sizes']) / (running_results['d_loss'] / running_results['batch_sizes']))
        results['learning_rate'].append(optimizerG.param_groups[0]['lr'])

        if epoch % 10 == 0:
            out_path = 'statistics/'
            os.makedirs(out_path, exist_ok=True)
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim'],
                      'Loss_Ratio': results['loss_ratio'], 'Learning_Rate': results['learning_rate']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + 'X_BN_train_results.csv', index_label='Epoch')





