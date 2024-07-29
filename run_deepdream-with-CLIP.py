import os
import sys
import clip
from clip.model import QuickGELU
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
import torchvision.utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
from argparse import Namespace
import random
import pdb
import collections
from typing import Any
import warnings
warnings.filterwarnings("ignore")
# Custom imports
from image_net import TotalVariation, CrossEntropyLoss, MatchBatchNorm, BaseFakeBN, LayerActivationNorm
from image_net import ActivationNorm, NormalVariation, ColorVariation, fix_random_seed
from image_net import NetworkPass
from image_net import LossArray, TotalVariation
from image_net import ViTFeatHook, ViTEnsFeatHook
from regularizers import TotalVariation as BaseTotalVariation, FakeColorDistribution as AbstractColorDistribution
from regularizers import FakeBatchNorm as BaseFakeBN, NormalVariation as BaseNormalVariation
from regularizers import ColorVariation as BaseColorVariation
from hooks import ViTAttHookHolder, ViTGeLUHook, ClipGeLUHook, SpecialSaliencyClipGeLUHook
from prepost import Clip, Tile, Jitter, RepeatBatch, ColorJitter, fix_random_seed
from prepost import GaussianNoise
from util import ClipWrapper
from util import new_init, fix_random_seed

parser = argparse.ArgumentParser(description="CLIP DeepDream")
parser.add_argument("--im", type=str, required=False, default="images/meshes.png", help="Input Image Path")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_nums = '0123456789'

steps_folder = 'steps'
os.makedirs(steps_folder, exist_ok=True)
steps_folder = 'finals'
os.makedirs(steps_folder, exist_ok=True)
steps_folder = 'dream'
os.makedirs(steps_folder, exist_ok=True)
debug_folder = 'debug'
os.makedirs(debug_folder, exist_ok=True)

filename = 'filename'

proc_folder = 'B_IMG_IN'
os.makedirs(proc_folder, exist_ok=True)


class ImageNetVisualizer:
    def __init__(self, loss_array: LossArray, pre_aug: nn.Module = None,
                 post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, save_every: int = 200, saver: bool = True,
                 print_every: int = 5, **_):
        self.loss = loss_array
        self.saver = saver
        print(self.saver)

        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None, layer: int = None, feature: int = None, clipname: str = None, stepfilename: str = None):
        if not img.is_cuda or img.device != torch.device('cuda:0'):
            img = img.to('cuda:0')
        if not img.requires_grad:
            img.requires_grad_()
        
        # ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']        
        # Default:
        # optimizer = optimizer if optimizer is not None else optim.Adam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        optimizer = optimizer if optimizer is not None else optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{self.loss.header()}', flush=True)

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f'{i}\t{self.loss}', flush=True)
            if i % self.save_every == 0 and self.saver is True:
                save_intermediate_step(img, i, layer, feature, clipname, stepfilename)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()

        optimizer.state = collections.defaultdict(dict)
        return img, stepfilename

def save_image(tensor: torch.Tensor, path: str):
    # If the tensor has a batch dimension, remove it
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    # Normalize the tensor to [0, 1] if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    torchvision.utils.save_image(tensor, path)

def save_intermediate_step(tensor: torch.Tensor, step: int, layer: int, feature: int, clipname: str, stepfilename: str, base_path='steps'):
    os.makedirs(base_path, exist_ok=True)
    base_path = f'steps/{clipname}_L{layer}-F{feature}/'
    os.makedirs(base_path, exist_ok=True)
    unique_identifier = stepfilename
    filename = f'{unique_identifier}_{step}.png'
    filepath = os.path.join(base_path, filename)

    # If the tensor has a batch dimension, remove it
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Normalize the tensor to [0, 1] if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    save_image(tensor, filepath)

def reassemble_tiles(tile_folder: str, final_folder: str):
    os.makedirs(final_folder, exist_ok=True)
    image_files = [f for f in os.listdir(tile_folder) if f.endswith(".png")]

    # Dictionary to group images
    grouped_images = {}
    for file in image_files:
        parts = file.split('_')
        tile_number = int(parts[0])
        key = '_'.join(parts[1:])
        if key not in grouped_images:
            grouped_images[key] = []
        grouped_images[key].append((tile_number, file))

    # Reassemble the images
    for key, tiles in grouped_images.items():
        # Sort tiles by the tile number
        tiles.sort()

        # Open tiles and arrange them
        tile_images = [Image.open(os.path.join(tile_folder, tile[1])) for tile in tiles]
        tile_width, tile_height = tile_images[0].size
        combined_image = Image.new('RGB', (2 * tile_width, 2 * tile_height))

        # Paste tiles into the combined image
        combined_image.paste(tile_images[0], (0, 0))
        combined_image.paste(tile_images[1], (tile_width, 0))
        combined_image.paste(tile_images[2], (0, tile_height))
        combined_image.paste(tile_images[3], (tile_width, tile_height))

        # Save the combined image
        combined_image.save(os.path.join(final_folder, key))

def get_clip_dimensions(clipmodel):
    model, preprocess = clip.load(clipmodel)
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            break
    num_layers = None
    num_features = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features
    return input_dims, num_layers, num_features

def load_clip_model(clipmodel, device='cuda'):
    model, _ = clip.load(clipmodel, device=device)
    model = ClipWrapper(model).to(device)
    return model

def generate_single(model, clipname, layer, feature, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed):
    loss = LossArray()
    loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient * tv)
    print(f"Debug: In generate loop: Layer:{layer} - Feature:{feature}")

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
    image = new_init(image_size, 1, use_fixed_random_seed=use_fixed_random_seed)

    visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient, use_fixed_random_seed=use_fixed_random_seed)
    image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

    save_image(image, f'finals/{clipname}_L{layer}_F{feature}.png')
    
def generate_deepdream(model, clipname, layer, feature, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed):
    loss = LossArray()
    loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient * tv)
    print(f"Debug: In generate loop: Layer:{layer} - Feature:{feature}")

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
    
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    folder_path = "B_IMG_IN"
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            img_saveloc = os.path.join('dream', img_name)

            image = Image.open(image_path)
            image = preprocess(image)
            image = image.unsqueeze(0).to(device)    
            
            stepfilename = img_name
            visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient, use_fixed_random_seed=use_fixed_random_seed)
            modified_image, filename = visualizer(image, layer=layer, feature=feature, clipname=clipname, stepfilename=stepfilename)
            image.data = modified_image.data
            save_image(image, f'{img_saveloc}_L{layer}_F{feature}.png')

    
class ClipNeuronCaptureHook:
    def __init__(self, module: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = None
        self.top_values = None
        self.top_indices = None
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.detach()

    def get_top_neurons(self, k=5):
        if self.activations is not None:
            self.top_values, self.top_indices = torch.topk(self.activations, k, dim=-1)
            return self.layer_idx, self.top_values, self.top_indices
        return None, None, None
    
    def remove(self):
        self.hook_handle.remove()

# Function to register hooks across all GELU layers
def register_hooks(model, num_layers):
    hooks = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, QuickGELU):
            hook = ClipNeuronCaptureHook(module, layer_idx)
            hooks.append(hook)
            layer_idx += 1
            if layer_idx >= num_layers:
                break
    return hooks

# After the forward pass
def get_all_top_neurons(hooks, k=5):
    all_top_neurons = []
    for hook in hooks:
        layer_idx, top_values, top_indices = hook.get_top_neurons(k)
        if top_values is not None:
            all_top_neurons.append((layer_idx, top_values, top_indices))
    return all_top_neurons

def clear_directory(directory: str):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def scale_and_tile_image(original_filename, input_dim, B_IMG_IN):
    # Clear the directory before saving new tiles
    clear_directory(B_IMG_IN)
    
    img = Image.open(original_filename).convert('RGB')

    # Scale the image to 2x model input dimensions
    new_size = (2 * input_dim, 2 * input_dim)
    img_resized = img.resize(new_size)
    
    # Calculate the size of each tile
    tile_width = new_size[0] // 2
    tile_height = new_size[1] // 2
    
    # Cut the image into four tiles and save each
    for i in range(2):
        for j in range(2):
            # Define the bounding box for the current tile
            left = i * tile_width
            upper = j * tile_height
            right = left + tile_width
            lower = upper + tile_height
            bbox = (left, upper, right, lower)
            
            # Crop the image to the bounding box to create the tile
            img_tile = img_resized.crop(bbox)
            
            # Construct the filename for the tile
            tile_filename = f"{i+j*2+1}_{os.path.basename(original_filename)}"
            
            # Save the tile to the specified directory
            img_tile.save(f"{B_IMG_IN}/{tile_filename}")

    
    
# CLIP models: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
clipmodel = 'ViT-L/14'
clipname = clipmodel.replace("/", "-").replace("@", "-")

input_dims, num_layers, num_features = get_clip_dimensions(clipmodel)
print(f"\nSelected input dimension for {clipmodel}: {input_dims}")
print(f"\nNumber of Layers: {num_layers} with {num_features} Features / Layer")

sideX = input_dims
sideY = input_dims

image_path = args.im
original_filename = image_path
dreamimage = image_path

transforming = transforms.Compose([
  transforms.Resize((sideX, sideY)),
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x[:3, :, :]),
  transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
  
def main():
    model = load_clip_model(clipmodel)
    image_size = input_dims
    generate_multi = True  # Set to True to visualize top_k (multiple) features for each layer
    
    # Load image and preprocess
    img = Image.open(image_path)
    input_image = transforming(img).unsqueeze(0).to(device)

    # Register hooks and perform forward pass
    hooks = register_hooks(model, num_layers)
    _ = model(input_image)
    
    # Retrieve top neurons across all layers
    all_top_neurons = get_all_top_neurons(hooks, k=5)
    top_features_per_layer = {}
    for layer_idx, _, top_indices in all_top_neurons:
        feature_indices = top_indices[0][0].cpu().tolist()
        top_features_per_layer[layer_idx] = feature_indices
    
    for hook in hooks:
        hook.remove()
        
    start_layer = num_layers-11 # 0 = input layer -> simplest patterns -> complex patterns -> output: final layer == (num_layers-1)
    end_layer = num_layers-5
    num_features_to_visualize = 2 # Number of top (activation value) features to visualize per layer, max. 5
    
    print(f"...Generating top {num_features_to_visualize} neuron activations for layers {start_layer} to {end_layer}...\n\n")
    
    # Visualization settings
    tv = 1.0
    coefficient=0.000005 # 0.0005: Soft, blurry; 0.00000005 = sharp, noisy; default = 0.000005 = balanced.
    lr = 0.1
    steps = 150
    print_every = 10
    save_every = 50
    saver = False  # Set to "False" to disable saving intermediate steps ("save_every") and only save final output image.
    use_fixed_random_seed = True

    # Generate visualizations for top features of selected layers
    if generate_multi:
        original_filename = image_path
        input_dim = input_dims  # Example input dimension
        B_IMG_IN = "B_IMG_IN"
        filename = "filename"
        scale_and_tile_image(original_filename, input_dim, B_IMG_IN)
        for layer_idx, feature_indices in top_features_per_layer.items():
            with open(f"debug/top-features.txt", "w", encoding='utf-8') as f:
                for layer_idx, feature_indices in top_features_per_layer.items():
                    f.write(f"Layer {layer_idx}: {feature_indices}\n")
        for layer_idx, feature_indices in top_features_per_layer.items():
            if start_layer <= layer_idx <= end_layer:  # Check if the layer is within the specified range
                # Limit the number of features to the first 'num_features_to_visualize'
                for feature_idx in feature_indices[:num_features_to_visualize]:
                    generate_deepdream(model, clipname, layer_idx, feature_idx, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed)
  
    else:
        # This part is for generating single visualization, set "generate_multi = False" 
        layer, feature = 20, 3169
        generate_single(model, clipname, layer, feature, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed)

    print("\nSaving full size images in the 'finals' folder...")
    reassemble_tiles('dream', 'finals')
    print("All done.")

if __name__ == '__main__':
    main()

    


