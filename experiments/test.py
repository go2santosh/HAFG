import matplotlib
matplotlib.use('Agg')  # headless: figures are only saved to files

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchprofile
from torch.utils.data import DataLoader
from torchvision import transforms

import copy
import json
import argparse

import os
import sys

# Run relative to this file so that "../data" and the project imports resolve from any working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from hafg_module import DehazeModel

from utils.test_functions import test, test_no_ref
from utils.config import load_config
from utils.dataset_loader import get_dataset_loader
from utils.iqa import PSNR, SSIM, LPIPS, FID, BRISQUE, NIQE, PIQE, FADE

# ---- Command line arguments ----
parser = argparse.ArgumentParser(description="Test a trained HAFG dehazing model")
parser.add_argument('--experiment-name', required=True, help="Experiment trained by train.py (output folder ../data/<dataset>_<experiment-name>)")
parser.add_argument('--dataset-name', default='nhhaze', help="Dataset the model was trained on")
parser.add_argument('--test-dataset-name', default=None, help="Dataset to test on (default: --dataset-name); rtts has no ground truth and is evaluated with no-reference metrics only")
parser.add_argument('--full-res', action='store_true', help="Test at the original image size instead of resizing to test_image_size (batch size 1; outputs go to *_fullres files)")
parser.add_argument('--checkpoint', default=None, help="Checkpoint file to test instead of the experiment's trained_model_checkpoint.pth (e.g. ../data/checkpoints/nhhaze.pth); without a config.json in the output folder, the default model settings are used")
parser.add_argument('--resize', dest='resize', choices=['bilinear', 'bicubic'], default=None, help="Filter used to resize the test images (default: resize of the run, bicubic)")
parser.add_argument('--test-size', dest='test_size', type=int, default=None, help="Side of the resized test images (default: test_image_size of the run, 256)")
args = parser.parse_args()

dataset_name = args.dataset_name
experiment_name = args.experiment_name
test_dataset_name = args.test_dataset_name or dataset_name
no_reference = test_dataset_name == 'rtts'
# Outputs of a cross-dataset test are tagged with the test dataset so they do not overwrite the in-domain results
file_tag = ('' if test_dataset_name == dataset_name else f'_{test_dataset_name}') + ('_fullres' if args.full_res else '')

# ---- Configuration ----
# Load the config written by the training run; outputs go next to it
output_dir = f"../data/{dataset_name}_{experiment_name}"
config_file_name = os.path.join(output_dir, 'config.json')
if os.path.exists(config_file_name):
    config = load_config(config_file_name)
    print("Config loaded from", config_file_name)
elif args.checkpoint is not None:
    # Standalone checkpoint: default model settings (the defaults of train.py)
    config = {'image_size': 256, 'test_image_size': 256, 'train_batch_size': 8, 'test_batch_size': 1}
    print("No config.json found; using the default model settings")
else:
    sys.exit(f"No config.json found in {output_dir}; train the model first or pass --checkpoint.")
os.makedirs(output_dir, exist_ok=True)
config['output_dir'] = output_dir
if args.test_size is not None:
    config['test_image_size'] = args.test_size
if args.resize is not None:
    config['resize'] = args.resize
config['device'] = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

# ---- Data loader setup ----
train_dataset, train_loader, eval_dataset, eval_loader, test_dataset, test_loader = get_dataset_loader(config, test_dataset_name)
print(f"Testing on {test_dataset_name} ({len(test_dataset)} images)")
if args.full_res:
    test_dataset.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---- Load trained model ----
model = DehazeModel(
    smarf_mid_channels=config.get('smarf_mid_channels', 92),
    guidance_channels=config.get('guidance_channels', 32),
    pgm_window=config.get('pgm_window', 7),
    pgm_p=config.get('pgm_p', 0.1),
    pgm_pool=config.get('pgm_pool', 'min'),
    base_channels=config.get('base_channels', 32),
    depth=config.get('depth', 2),
    omega=config.get('omega', 0.90),
    omega_learnable=config.get('omega_learnable', True),
    use_pgm=config.get('use_pgm', True),
    use_fafm=config.get('use_fafm', True),
    use_safr=config.get('use_safr', True),
    use_smarf=config.get('use_smarf', True),
    use_chroma=config.get('use_ccfm', True)
).to(config['device'])
trained_model_checkpoint_path = args.checkpoint or os.path.join(config['output_dir'], 'trained_model_checkpoint.pth')
if not os.path.exists(trained_model_checkpoint_path):
    sys.exit(f"No checkpoint found at {trained_model_checkpoint_path}; train the model first.")
model_checkpoint = torch.load(trained_model_checkpoint_path, map_location=config['device'], weights_only=False)
# Load only the model weights
load_result = model.load_state_dict(model_checkpoint['model'], strict=False)
model.eval()  # set to inference mode
print(f"Loaded model from checkpoint (trained for {model_checkpoint.get('epoch')} epochs).")
if load_result.missing_keys or load_result.unexpected_keys:
    print(f"[Warning] Missing keys: {load_result.missing_keys}; unexpected keys: {load_result.unexpected_keys}")


class PaddedModel(nn.Module):
    """Reflect-pads the input to a multiple of 32 (the network needs sizes divisible by its down-sampling), runs the model and crops the output back."""
    def __init__(self, model, multiple=32):
        super().__init__()
        self.model = model
        self.multiple = multiple

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % self.multiple, (-width) % self.multiple
        out = self.model(F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else x)
        return out[..., :height, :width]

if args.full_res:
    model = PaddedModel(model)

# Check that the model returns the expected output in [-1, 1] (the RTTS loader returns hazy images only)
with torch.no_grad():
    first_batch = next(iter(test_loader))
    hazy_batch = (first_batch if no_reference else first_batch[0]).to(config['device'])
    restored_batch = model(hazy_batch)
if not no_reference:
    print("Min/Max of clear_batch:", first_batch[1].min().item(), first_batch[1].max().item())
print("Min/Max of hazy_batch:", hazy_batch.min().item(), hazy_batch.max().item())
print("Min/Max of model output:", restored_batch.min().item(), restored_batch.max().item())

# ---- Calculate model's FLOPs and parameters ----
params = sum(p.numel() for p in model.parameters())
inp = torch.randn(1, 3, 256, 256)
try:
    macs = torchprofile.profile_macs(model, inp.to(config['device']))
except Exception as e:
    print(f"[Warning] MAC count on {config['device']} failed ({e}); retrying on cpu")
    try:
        macs = torchprofile.profile_macs(copy.deepcopy(model).to('cpu'), inp)
    except Exception as e:
        print(f"[Warning] MAC count failed ({e}); not reported")
        macs = None

if macs is not None:
    print(f"MACs: {macs / 1e9:.3f} GMACs")
print(f"Params: {params / 1e6:.3f} M")

# ---- Report PSNR, SSIM, FID, LPIPS and no-reference metrics on the test dataset ----
# Create metrics functions
psnr_fn = PSNR(data_range=2.0)
ssim_fn = SSIM
lpips_fn = LPIPS.to(config['device'])
brisque_fn = BRISQUE
niqe_fn = NIQE
piqe_fn = PIQE
fade_fn = FADE()
fid_fn = FID.to('cpu')

if no_reference:
    # No ground truth: BRISQUE, NIQE, PIQE and FADE of the restored images only
    best_values = test_no_ref(model, test_loader, config['device'],
        config,
        brisque_fn=brisque_fn,
        niqe_fn=niqe_fn,
        piqe_fn=piqe_fn,
        fade_fn=fade_fn,
        file_prefix=f'trained_comparison{file_tag}', only_top_1=True, save_all=True, show_plt=False)
else:
    best_values = test(model, test_loader, config['device'],
        config,
        psnr_fn=psnr_fn,
        ssim_fn=ssim_fn,
        lpips_fn=lpips_fn.to(config['device']),
        brisque_fn=brisque_fn,
        niqe_fn=niqe_fn,
        piqe_fn=piqe_fn,
        fade_fn=fade_fn,
        fid_fn=fid_fn,
        file_prefix=f'trained_comparison{file_tag}', only_top_1=True, save_all=True, show_plt=False)

# Append FLOPs and parameters to the best_values
best_values['FLOPs'] = f"{macs / 1e9} G" if macs is not None else None
best_values['Parameters'] = f"{params / 1e6} M"

# Mean over the test images (the top level metrics above are the best single image), plus what this run was trained with
metric_names = ['brisque', 'niqe', 'piqe', 'fade'] if no_reference else ['psnr', 'ssim', 'lpips', 'brisque', 'niqe', 'piqe', 'fade']
details = best_values['details']
best_values['mean'] = {name: sum(d[name] for d in details) / len(details) for name in metric_names}
if not no_reference:
    best_values['mean']['fid'] = best_values['fid']
best_values['test_dataset'] = test_dataset_name
best_values['checkpoint_epoch'] = model_checkpoint.get('epoch')
best_values['protocol'] = 'full resolution (reflect-padded to a multiple of 32)' if args.full_res else f"resized to {config['test_image_size']} x {config['test_image_size']} ({config.get('resize', 'bicubic')})"
best_values['settings'] = {key: config.get(key) for key in (
    'lambda_l1', 'lambda_ssim', 'lambda_freq_amp', 'lambda_freq_phase', 'lambda_color',
    'omega', 'omega_learnable', 'use_pgm', 'use_fafm', 'use_safr', 'use_smarf', 'use_ccfm', 'seed',
    'smarf_mid_channels', 'guidance_channels', 'pgm_window', 'pgm_p', 'pgm_pool', 'base_channels', 'depth', 'lr', 'weight_decay',
    'shuffle', 'augment', 'train_subset', 'resize')}

# Save best values to a file
results_file = os.path.join(config['output_dir'], f'test_best_values{file_tag}.txt')
with open(results_file, 'w') as f:
    json.dump(best_values, f, indent=4, separators=(',', ': '))
print("Test results saved to", results_file)
print("Mean over test images:", best_values['mean'])
