import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

# Compare clear image, hazy image, and restored clear image
# Show comparison_results in the title
def show_comparison(hazy_image_np, restored_image_np, clear_image_np, metrics_label, save_to_files=[], show_plt=True):
    fig = plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.title("Hazy Image")
    plt.imshow(np.clip(hazy_image_np, 0, 1))
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Restored Clear Image')
    plt.imshow(np.clip(restored_image_np, 0, 1))
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Original Clear Image")
    plt.imshow(np.clip(clear_image_np, 0, 1))
    plt.axis('off')
    plt.figtext(0.5, 0.01, metrics_label, ha="center", fontsize=12)

    for save_to_file in save_to_files:
        plt.savefig(save_to_file, bbox_inches='tight')
        print(f"Saved to : {save_to_file}")
    
    if show_plt:
        plt.show()
    plt.close(fig)
        
def test(model, test_loader, device, 
        config,
        psnr_fn,
        ssim_fn,
        lpips_fn,
        brisque_fn,
        niqe_fn,
        piqe_fn,
        fade_fn,
        fid_fn,
        file_prefix='comparison', only_top_1=True, save_all=False, show_plt=True):
    
    model.eval()
    best_values = {'psnr':-1e10, 'ssim':-1e10, 'lpips':1e10, 'fid':1e10, 'brisque':1e10, 'niqe':1e10, 'piqe':1e10, 'fade':1e10, 'details':[]}
    first_best_ssim = -1e10
    first_best_psnr = -1e10
    first_best_lpips = 1e10
    first_best_brisque = 1e10
    first_best_niqe = 1e10
    first_best_piqe = 1e10
    first_best_fade = 1e10
    image_index = 0
    
    n = 0
    # Max n
    max_n = len(test_loader) / config.get("test_random_skips", 1)

    with torch.no_grad():
        for hazy_batch, clear_batch in tqdm(test_loader, desc="Testing"):

            # Exit if exceed max
            if n >= max_n: 
                break
            n += 1

            hazy_batch = hazy_batch.to(device); clear_batch = clear_batch.to(device)
            restored_batch = model(hazy_batch)
            batch_size = hazy_batch.shape[0]

            for i in range(batch_size):
                hazy_image = hazy_batch[i].cpu()
                clear_image = clear_batch[i].cpu()
                restored_image = restored_batch[i].cpu()

                # Compute psnr and ssim and lpips for this image while in [-1,1]
                psnr_val = psnr_fn(restored_image.unsqueeze(0), clear_image.unsqueeze(0)).item()
                ssim_val = ssim_fn(restored_image.unsqueeze(0), clear_image.unsqueeze(0), data_range=2.0).item()
                restored_image_lpips = restored_image.unsqueeze(0).to(device)
                clear_image_lpips = clear_image.unsqueeze(0).to(device)
                lpips_val = lpips_fn(restored_image_lpips, clear_image_lpips).item()

                # Denormalize from [-1,1] to [0,1]
                hazy_image_01 = (hazy_image * 0.5) + 0.5
                clear_image_01 = (clear_image * 0.5) + 0.5
                restored_image_01 = (restored_image * 0.5) + 0.5

                # No-Reference IQA metrics in [0,1]
                brisque_val = brisque_fn(restored_image_01.unsqueeze(0)).item()
                niqe_val = niqe_fn(restored_image_01.unsqueeze(0)).item()
                piqe_val = piqe_fn(restored_image_01.unsqueeze(0)).item()
                fade_val = fade_fn(restored_image_01.unsqueeze(0)).item()

                # FID expects [N, 3, 299, 299] in [0,1]
                restored_image_fid = F.interpolate(restored_image_01.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False)
                clear_image_fid = F.interpolate(clear_image_01.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False)
                fid_fn.update(torch.clamp(restored_image_fid, 0, 1), real=False)
                fid_fn.update(torch.clamp(clear_image_fid, 0, 1), real=True)

                # Update best values
                best_values['psnr'] = max(best_values['psnr'], psnr_val)
                best_values['ssim'] = max(best_values['ssim'], ssim_val)
                best_values['lpips'] = min(best_values['lpips'], lpips_val)
                best_values['brisque'] = min(best_values['brisque'], brisque_val)
                best_values['niqe'] = min(best_values['niqe'], niqe_val)
                best_values['piqe'] = min(best_values['piqe'], piqe_val)
                best_values['fade'] = min(best_values['fade'], fade_val)

                # Metrics label
                metrics_label = f'PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}'
                metrics_label = f'{metrics_label}, BRISQUE: {brisque_val:.4f}'
                metrics_label = f'{metrics_label}, NIQE: {niqe_val:.4f}'
                metrics_label = f'{metrics_label}, PIQE: {piqe_val:.4f}'
                metrics_label = f'{metrics_label}, FADE: {fade_val:.4f}'
                
                save_to_files = []
                if only_top_1:
                    if psnr_val > first_best_psnr:
                        first_best_psnr = psnr_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_psnr.png')
                        save_to_files.append(save_to_file)
                    if ssim_val > first_best_ssim:
                        first_best_ssim = ssim_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_ssim.png')
                        save_to_files.append(save_to_file)
                    if lpips_val < first_best_lpips:
                        first_best_lpips = lpips_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_lpips.png')
                        save_to_files.append(save_to_file)
                    if brisque_val < first_best_brisque:
                        first_best_brisque = brisque_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_brisque.png')
                        save_to_files.append(save_to_file)
                    if niqe_val < first_best_niqe:
                        first_best_niqe = niqe_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_niqe.png')
                        save_to_files.append(save_to_file)
                    if piqe_val < first_best_piqe:
                        first_best_piqe = piqe_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_piqe.png')
                        save_to_files.append(save_to_file)
                    if fade_val < first_best_fade:
                        first_best_fade = fade_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_fade.png')
                        save_to_files.append(save_to_file)
                if save_all:
                    save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_idx{image_index}.png')
                    save_to_files.append(save_to_file)

                # Visualize the results
                show_comparison(hazy_image_01.numpy().transpose(1,2,0), 
                                restored_image_01.numpy().transpose(1,2,0), 
                                clear_image_01.numpy().transpose(1,2,0), 
                                metrics_label, 
                                save_to_files=save_to_files,
                                show_plt=show_plt)
                image_index += 1

                # Add comparison details
                best_values['details'].append(
                    {'index':n+i, 
                     'psnr':psnr_val, 
                     'ssim':ssim_val, 
                     'lpips':lpips_val,
                     'brisque':brisque_val,
                     'niqe':niqe_val,
                     'piqe':piqe_val,
                     'fade':fade_val
                     })
        # Compute FID    
        best_values['fid'] = fid_fn.compute().item()
    return best_values

def show_comparison_no_ref(hazy_image_np, restored_image_np, metrics_label, save_to_files=[], show_plt=True):
    fig = plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Hazy Image")
    plt.imshow(np.clip(hazy_image_np, 0, 1))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Restored Clear Image')
    plt.imshow(np.clip(restored_image_np, 0, 1))
    plt.axis('off')
    plt.figtext(0.5, 0.01, metrics_label, ha="center", fontsize=12)

    for save_to_file in save_to_files:
        plt.savefig(save_to_file, bbox_inches='tight')
        print(f"Saved to : {save_to_file}")
    
    if show_plt:
        plt.show()
    plt.close(fig)

def test_no_ref(model, test_loader, device, 
        config,
        brisque_fn,
        niqe_fn,
        piqe_fn,
        fade_fn,
        file_prefix='comparison', only_top_1=True, save_all=False, show_plt=True):
    
    model.eval()
    best_values = {'brisque':1e10, 'niqe':1e10, 'piqe':1e10, 'fade':1e10, 'details':[]}
    first_best_brisque = 1e10
    first_best_niqe = 1e10
    first_best_piqe = 1e10
    first_best_fade = 1e10
    image_index = 0
    
    n = 0
    # Max n
    max_n = len(test_loader) / config.get("test_random_skips", 1)

    with torch.no_grad():
        for image_batch in tqdm(test_loader, desc="Testing"):

            # Exit if exceed max
            if n >= max_n: 
                break
            n += 1

            image_batch = image_batch.to(device)
            restored_batch = model(image_batch)
            batch_size = image_batch.shape[0]

            for i in range(batch_size):
                input_image = image_batch[i].cpu()
                restored_image = restored_batch[i].cpu()

                # Denormalize from [-1,1] to [0,1]
                input_image_01 = (input_image * 0.5) + 0.5
                restored_image_01 = (restored_image * 0.5) + 0.5

                # No-Reference IQA metrics in [0,1]
                brisque_val = brisque_fn(restored_image_01.unsqueeze(0)).item()
                niqe_val = niqe_fn(restored_image_01.unsqueeze(0)).item()
                piqe_val = piqe_fn(restored_image_01.unsqueeze(0)).item()
                fade_val = fade_fn(restored_image_01.unsqueeze(0)).item()

                # Update best values
                best_values['brisque'] = min(best_values['brisque'], brisque_val)
                best_values['niqe'] = min(best_values['niqe'], niqe_val)
                best_values['piqe'] = min(best_values['piqe'], piqe_val)
                best_values['fade'] = min(best_values['fade'], fade_val)

                # Metrics label
                metrics_label = f'BRISQUE: {brisque_val:.4f}'
                metrics_label = f'{metrics_label}, NIQE: {niqe_val:.4f}'
                metrics_label = f'{metrics_label}, PIQE: {piqe_val:.4f}'
                metrics_label = f'{metrics_label}, FADE: {fade_val:.4f}'
                
                save_to_files = []
                if only_top_1:
                    if brisque_val < first_best_brisque:
                        first_best_brisque = brisque_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_brisque.png')
                        save_to_files.append(save_to_file)
                    if niqe_val < first_best_niqe:
                        first_best_niqe = niqe_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_niqe.png')
                        save_to_files.append(save_to_file)
                    if piqe_val < first_best_piqe:
                        first_best_piqe = piqe_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_piqe.png')
                        save_to_files.append(save_to_file)
                    if fade_val < first_best_fade:
                        first_best_fade = fade_val
                        save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_1st_best_fade.png')
                        save_to_files.append(save_to_file)
                if save_all:
                    save_to_file = os.path.join(config['output_dir'], f'{file_prefix}_idx{image_index}.png')
                    save_to_files.append(save_to_file)

                # Visualize the results
                show_comparison_no_ref(input_image_01.numpy().transpose(1,2,0), 
                                restored_image_01.numpy().transpose(1,2,0), 
                                metrics_label, 
                                save_to_files=save_to_files,
                                show_plt=show_plt)
                image_index += 1

                # Add comparison details
                best_values['details'].append(
                    {'index':n+i, 
                     'brisque':brisque_val,
                     'niqe':niqe_val,
                     'piqe':piqe_val,
                     'fade':fade_val
                     })
                
    return best_values
