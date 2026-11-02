import torch

import json
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import argparse

# Run relative to this file so that "../data" and the project imports resolve from any working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")

from utils.config import save_config, load_config
from utils.dataset_loader import get_dataset_loader
from hafg_module import DehazeModel
from utils.loss_functions import LossFunctions

# ---- Command line arguments ----
DEFAULT_LAMBDAS = {
    'lambda_l1': 1.0,
    'lambda_ssim': 0.5,
    'lambda_freq_amp': 0.01,
    'lambda_freq_phase': 0.1,
    'lambda_color': 0.01
}

# Random seed (weight initialisation and data order)
DEFAULT_TRAIN_PARAMS = {
    'seed': 42,
    # Training recipe
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'shuffle': True,
    'augment': True,
    'train_subset': 0,
    'resize': 'bicubic'
}

DEFAULT_MODEL_PARAMS = {
    # Components (PGM = prior-guided skip connections, CCFM = chroma branch); True = enabled
    'use_pgm': True,
    'use_fafm': True,
    'use_safr': True,
    'use_smarf': True,
    'use_ccfm': True,
    # Omega of the transmission t = 1 - omega * D (initial value if learnable)
    'omega': 0.90,
    'omega_learnable': True,
    # Network width and depth
    'base_channels': 32,
    'depth': 2,
    'smarf_mid_channels': 92,
    'guidance_channels': 32,
    'pgm_window': 7,
    'pgm_p': 0.1,
    'pgm_pool': 'min'
}

parser = argparse.ArgumentParser(description="Train the HAFG dehazing model")
parser.add_argument('--experiment-name', required=True, help="Output goes to ../data/<dataset>_<experiment-name>; use a distinct name per ablation run")
parser.add_argument('--dataset-name', default='nhhaze')
parser.add_argument('--source-images-path', default=None, help="Subset json (default: ../data/<dataset>_<experiment-name>/subset.json; the whole dataset is used if the file is missing)")
for key, value in DEFAULT_LAMBDAS.items():
    parser.add_argument('--' + key.replace('_', '-'), dest=key, type=float, default=None,
        help=f"Loss weight (default {value}; on resume, the value saved in config.json)")
for key in ('use_pgm', 'use_fafm', 'use_safr', 'use_smarf', 'use_ccfm'):
    flag = key.replace('use_', '')
    parser.add_argument('--' + flag, dest=key, action=argparse.BooleanOptionalAction, default=None,
        help=f"Enable/disable {flag.upper()} (--{flag} / --no-{flag}; default enabled; on resume, the value saved in config.json)")
parser.add_argument('--epochs', type=int, default=None, help="Total number of epochs (default: value in config.json, or 250 for a new experiment)")
parser.add_argument('--seed', type=int, default=None,
    help=f"Random seed for weight initialisation and data order (default {DEFAULT_TRAIN_PARAMS['seed']}; on resume, the value saved in config.json)")
parser.add_argument('--omega', type=float, default=None,
    help=f"Omega of the transmission t = 1 - omega * D (default {DEFAULT_MODEL_PARAMS['omega']}; initial value when learnable; on resume, the value saved in config.json)")
parser.add_argument('--smarf-mid-channels', dest='smarf_mid_channels', type=int, default=None, help="Width of the SMARF branch convolutions (default 92)")
parser.add_argument('--guidance-channels', dest='guidance_channels', type=int, default=None, help="Channels of the prior guidance G' of every skip connection (default 32)")
parser.add_argument('--pgm-pool', dest='pgm_pool', choices=['min', 'max', 'mean'], default=None, help="PGM: pooling over the feature channels for the haze-density map (default min)")
parser.add_argument('--pgm-window', dest='pgm_window', type=int, default=None, help="PGM: side of the local patch of the haze-density map (default 7, odd)")
parser.add_argument('--pgm-p', dest='pgm_p', type=float, default=None, help="PGM: fraction p of the lowest values of the haze-density map that is averaged to the haze strength (default 0.1)")
parser.add_argument('--base-channels', dest='base_channels', type=int, default=None, help="Width of the first level (default 32); the other levels double it")
parser.add_argument('--depth', dest='depth', type=int, default=None, help="Number of encoder and decoder levels (default 2)")
parser.add_argument('--lr', dest='lr', type=float, default=None, help="Initial learning rate (default 1e-4)")
parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=None, help="AdamW weight decay (default 1e-6)")
parser.add_argument('--shuffle', dest='shuffle', action=argparse.BooleanOptionalAction, default=None, help="Shuffle the training images every epoch (default on)")
parser.add_argument('--augment', dest='augment', action=argparse.BooleanOptionalAction, default=None, help="Also random vertical flips and 90-degree rotations of the training crops (default on)")
parser.add_argument('--resize', dest='resize', choices=['bilinear', 'bicubic'], default=None, help="Filter used to resize the images to the training size (default bicubic)")
parser.add_argument('--train-subset', dest='train_subset', type=int, default=None, help="Train on this many randomly selected training images (fixed by --seed; default 0 = all), e.g. 1000 for OTS")
parser.add_argument('--omega-learnable', dest='omega_learnable', action=argparse.BooleanOptionalAction, default=None,
    help="Train omega (--omega-learnable, default) or keep it fixed (--no-omega-learnable)")
args = parser.parse_args()


device = "cpu"
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type

print(f"Using {device} device")

dataset_name = args.dataset_name
experiment_name = args.experiment_name
config = {
        'source_images_path': args.source_images_path or f"../data/{dataset_name}_{experiment_name}/subset.json",
        'output_dir': f"../data/{dataset_name}_{experiment_name}",
        'model_checkpoint_path': f"../data/{dataset_name}_{experiment_name}/trained_model_checkpoint.pth",
        'image_size': 256,
        'test_image_size': 256,
        'device': device,
        'train_batch_size': 8,
        'test_batch_size': 1,
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'epochs': 250,
        'cosine_t_max': 250,  # length of each cosine cycle (epochs); LR restarts every cycle
        'eta_min': 1e-6
    }

if not os.path.exists(config['output_dir']):
    os.makedirs(config['output_dir'])

config_file_name = os.path.join(config['output_dir'], 'config.json')
if not os.path.exists(config_file_name):
    save_config(config_file_name, config)
    print("Config saved to", config_file_name)
else:
    config = load_config(config_file_name)
    print("Config loaded from", config_file_name)

config.setdefault('eta_min', 1e-6)
if args.epochs is not None:
    config['epochs'] = args.epochs

# Loss weights, model parameters and seed: command line > value saved in config.json (resume) > default.
# Stored in config.json so every run records what it was trained with.
for key, default in {**DEFAULT_LAMBDAS, **DEFAULT_MODEL_PARAMS, **DEFAULT_TRAIN_PARAMS}.items():
    cli_value = getattr(args, key)
    saved_value = config.get(key)
    if cli_value is not None and saved_value is not None and cli_value != saved_value:
        print(f"[Warning] {key}={cli_value} differs from {saved_value} saved in config.json; using {cli_value}")
    config[key] = cli_value if cli_value is not None else (saved_value if saved_value is not None else default)
save_config(config_file_name, config)
print("Loss weights:", {key: config[key] for key in DEFAULT_LAMBDAS})
print("Model parameters:", {key: config[key] for key in DEFAULT_MODEL_PARAMS})

# (On resume the saved RNG state in the checkpoint takes over, so the random stream continues exactly.)
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
print("Random seed:", config['seed'])


source_images = None
if os.path.exists(config['source_images_path']):
    with open(config['source_images_path'], 'r') as file:
        subset_json_data = json.load(file)
        source_images = [item['source_image_name'] for item in subset_json_data]

train_dataset, train_loader, eval_dataset, eval_loader, test_dataset, test_loader = get_dataset_loader(config, dataset_name, source_images=source_images)

if train_dataset is not None and (config['augment'] or config['shuffle']):
    from torch.utils.data import DataLoader, Dataset

    class AugmentedPairs(Dataset):
        """Random vertical flip and 90-degree rotation, applied identically to the hazy and the clear crop."""
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, index):
            hazy, clear = self.dataset[index][:2]
            if random.random() < 0.5:
                hazy, clear = hazy.flip(-2), clear.flip(-2)
            turns = random.randint(0, 3)
            if turns:
                hazy, clear = torch.rot90(hazy, turns, (-2, -1)), torch.rot90(clear, turns, (-2, -1))
            return hazy, clear

    if config['augment']:
        train_dataset = AugmentedPairs(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=config['shuffle'])
    print(f"Training data: {len(train_dataset)} images per epoch, shuffle={config['shuffle']}, augment={config['augment']}")

model = DehazeModel(
    base_channels=config['base_channels'],
    depth=config['depth'],
    omega=config['omega'],
    omega_learnable=config['omega_learnable'],
    use_pgm=config['use_pgm'],
    use_fafm=config['use_fafm'],
    use_safr=config['use_safr'],
    use_smarf=config['use_smarf'],
    use_chroma=config['use_ccfm'],
    smarf_mid_channels=config['smarf_mid_channels'],
    guidance_channels=config['guidance_channels'],
    pgm_window=config['pgm_window'],
    pgm_p=config['pgm_p'],
    pgm_pool=config['pgm_pool']
).to(device)
if os.path.exists(config['model_checkpoint_path']):
    model_checkpoint = torch.load(config['model_checkpoint_path'], map_location=device, weights_only=False)
    result = model.load_state_dict(model_checkpoint['model'], strict=False)
    print("Loaded model from checkpoint.")
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)
else:
    print("No checkpoint found, starting from scratch.")

class HAFGTrainer:
    def __init__(self, config, train_loader, eval_loader, model):
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model

        self.device = config['device']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        # Cosine annealing with warm restarts: LR decays to eta_min over cosine_t_max epochs, then jumps back to lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config['cosine_t_max'], eta_min=config['eta_min'])
        self.model_checkpoint_path = config['model_checkpoint_path']

        # Exact resume: restore optimizer, scheduler and RNG state when the checkpoint has them
        self.resume_epoch = None
        if self.model_checkpoint_path is not None and os.path.exists(self.model_checkpoint_path):
            ckpt = torch.load(self.model_checkpoint_path, map_location=self.device, weights_only=False)
            if 'scheduler' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.scheduler.load_state_dict(ckpt['scheduler'])
                if 'rng_state' in ckpt:
                    torch.set_rng_state(ckpt['rng_state'].cpu())
                    np.random.set_state(ckpt['numpy_rng_state'])
                    random.setstate(ckpt['python_rng_state'])
                self.resume_epoch = int(ckpt['epoch'])
                print(f"Restored optimizer/scheduler state; resuming after epoch {self.resume_epoch}")
            else:
                print("[Warning] Checkpoint has no scheduler state (older format); optimizer state starts fresh.")
        self.loss_functions = LossFunctions()

        self.l1_loss_fn = self.loss_functions.l1_loss_fn
        self.ssim_loss_fn = self.loss_functions.ssim_loss_fn
        self.freq_amp_loss_fn = self.loss_functions.freq_amp_loss_fn
        self.freq_phase_loss_fn = self.loss_functions.freq_phase_loss_fn
        self.color_loss_fn = self.loss_functions.color_loss_fn

    @torch.no_grad()
    def eval(self, 
        lambda_l1,
        lambda_ssim,
        lambda_freq_amp,
        lambda_freq_phase,
        lambda_color):
        n = 0
        self.model.eval()
        eval_loss = {
            'total_loss': 0, 
            'l1_loss': 0, 
            'ssim_loss': 0, 
            'freq_amp_loss': 0, 
            'freq_phase_loss': 0,
            'color_loss': 0
        }
        for hazy_batch, clear_batch in self.eval_loader:
            hazy_batch, clear_batch = hazy_batch.to(device), clear_batch.to(device)
            restored_batch = self.model(hazy_batch)

            l1_loss = self.l1_loss_fn(restored_batch, clear_batch)
            ssim_loss = 1.0 - self.ssim_loss_fn(restored_batch, clear_batch, data_range=2.0, size_average=True)
            freq_amp_loss = self.freq_amp_loss_fn(restored_batch, clear_batch)
            freq_phase_loss = self.freq_phase_loss_fn(restored_batch, clear_batch)
            color_loss = self.color_loss_fn(restored_batch.clamp(-1.0, 1.0), clear_batch)  # the Lab conversion gives NaN gradients outside the image range

            loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss + lambda_freq_amp * freq_amp_loss + lambda_freq_phase * freq_phase_loss + lambda_color * color_loss

            eval_loss['total_loss'] += loss.item()
            eval_loss['l1_loss'] += l1_loss.item()
            eval_loss['ssim_loss'] += ssim_loss.item()
            eval_loss['freq_amp_loss'] += freq_amp_loss.item()
            eval_loss['freq_phase_loss'] += freq_phase_loss.item()
            eval_loss['color_loss'] += color_loss.item()

            n += 1
        
        for k in eval_loss:
            eval_loss[k] /= max(n, 1)

        return eval_loss

    def train(self,
        epochs, 
        lambda_l1,
        lambda_ssim,
        lambda_freq_amp,
        lambda_freq_phase,
        lambda_color):

        # Completed epochs: from the checkpoint when it carries full state, else from config
        if self.resume_epoch is not None:
            start_epoch = self.resume_epoch
        else:
            start_epoch = int(config.get("last_epoch", 0))
        print(f"Starting training from epoch {start_epoch+1}")

        writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'logs'))

        # Legacy checkpoints have no scheduler state: fast-forward the scheduler instead
        if self.resume_epoch is None:
            for _ in range(start_epoch):
                self.scheduler.step()

        for epoch in range(start_epoch, epochs):
            n = 0
            self.model.train()
            train_loss = {
                'total_loss': 0, 
                'l1_loss': 0, 
                'ssim_loss': 0, 
                'freq_amp_loss': 0, 
                'freq_phase_loss': 0,
                'color_loss': 0
            }
            for hazy_batch, clear_batch in self.train_loader:
                hazy_batch, clear_batch = hazy_batch.to(device), clear_batch.to(device)
                self.optimizer.zero_grad()

                # Forward pass
                restored_batch = self.model(hazy_batch)

                l1_loss = self.l1_loss_fn(restored_batch, clear_batch)
                ssim_loss = 1.0 - self.ssim_loss_fn(restored_batch, clear_batch, data_range=2.0, size_average=True)
                freq_amp_loss = self.freq_amp_loss_fn(restored_batch, clear_batch)
                freq_phase_loss = self.freq_phase_loss_fn(restored_batch, clear_batch)
                color_loss = self.color_loss_fn(restored_batch.clamp(-1.0, 1.0), clear_batch)  # the Lab conversion gives NaN gradients outside the image range

                loss = lambda_l1 * l1_loss + lambda_ssim * ssim_loss + lambda_freq_amp * freq_amp_loss + lambda_freq_phase * freq_phase_loss + lambda_color * color_loss

                # Backward pass
                loss.backward()

                # Gradient check and step
                valid_gradients = True
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            break
                if valid_gradients:
                    self.optimizer.step()
                else:
                    print("[Warning] NaN/Inf detected in gradients. Skipping batch parameter update.")
                    sys.stdout.flush()

                train_loss['total_loss'] += loss.item()
                train_loss['l1_loss'] += l1_loss.item()
                train_loss['ssim_loss'] += ssim_loss.item()
                train_loss['freq_amp_loss'] += freq_amp_loss.item()
                train_loss['freq_phase_loss'] += freq_phase_loss.item()
                train_loss['color_loss'] += color_loss.item()

                n += 1
            
            for k in train_loss:
                train_loss[k] /= max(n, 1)

            GREEN = '\033[92m'
            RESET = '\033[0m'
            print(f"{GREEN}Epoch {epoch+1} Training: total_loss={train_loss['total_loss']:.4f}, l1_loss={train_loss['l1_loss']:.4f}, ssim_loss={train_loss['ssim_loss']:.4f}, freq_amp_loss;{train_loss['freq_amp_loss']:.4f}, freq_phase_loss:{train_loss['freq_phase_loss']:.4f}, color_loss:{train_loss['color_loss']:.4f}{RESET}")

            eval_loss = self.eval(lambda_l1, lambda_ssim, lambda_freq_amp, lambda_freq_phase, lambda_color)
            print(f"Epoch {epoch+1} Evaluation: total_loss={eval_loss['total_loss']:.4f}, l1_loss={eval_loss['l1_loss']:.4f}, ssim_loss={eval_loss['ssim_loss']:.4f}, freq_amp_loss;{eval_loss['freq_amp_loss']:.4f}, freq_phase_loss:{eval_loss['freq_phase_loss']:.4f}, color_loss:{eval_loss['color_loss']:.4f}")

            writer.add_scalars('Metrics', {
                'train_loss': train_loss['total_loss'],
                'eval_loss': eval_loss['total_loss'],
                'train_l1_loss': train_loss['l1_loss'],
                'train_ssim_loss': train_loss['ssim_loss'],
                'train_freq_amp_loss': train_loss['freq_amp_loss'],
                'train_freq_phase_loss': train_loss['freq_phase_loss'],
                'train_color_loss': train_loss['color_loss'],
                'eval_l1_loss': eval_loss['l1_loss'],
                'eval_ssim_loss': eval_loss['ssim_loss'],
                'eval_freq_amp_loss': eval_loss['freq_amp_loss'],
                'eval_freq_phase_loss': eval_loss['freq_phase_loss'],
                'eval_color_loss': eval_loss['color_loss']
            }, epoch)
            writer.flush()

            # Step the cosine scheduler once per epoch (restarts every cosine_t_max epochs)
            self.scheduler.step()
            writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Save model checkpoint (after the scheduler step so the saved state is consistent)
            if self.model_checkpoint_path is not None:
                torch.save({
                    'epoch': epoch + 1,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'python_rng_state': random.getstate()
                }, self.model_checkpoint_path)

            config['last_epoch'] = epoch+1
            save_config(os.path.join(config['output_dir'], "config.json"), config)
        
        writer.close()
        print("Training complete.")

hafg_trainer = HAFGTrainer(
    config=config,
    train_loader=train_loader,
    eval_loader=eval_loader,
    model=model
    )

hafg_trainer.train(epochs=config['epochs'],
        lambda_l1=config['lambda_l1'],
        lambda_ssim=config['lambda_ssim'],
        lambda_freq_amp=config['lambda_freq_amp'],
        lambda_freq_phase=config['lambda_freq_phase'],
        lambda_color=config['lambda_color'])
