import torch
import os
from longclipmodel import longclip
from longclipmodel.model_longclip import QuickGELU
import longclipmodel.simple_tokenizer as simpletokenizer
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import kornia
import argparse
from colorama import Fore, Style
from torch.cuda.amp import autocast, GradScaler
import warnings
from prepost import Clip, Tile, Jitter, RepeatBatch, ColorJitter
from prepost import GaussianNoise
from natsort import natsorted
warnings.filterwarnings('ignore')
scaler = GradScaler()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="CLIP Full Model DeepDream")
parser.add_argument("--im", type=str, required=False, default="images/beachscene.png", help="Input Image Path")
args = parser.parse_args()


# ++ General CLIP model settings ++
    
# https://github.com/beichenzbc/Long-CLIP and download the checkpoint(s) to use here:    
clipmodel='path/to/checkpoints/longclip-L.pt'
checkin_step = 10 # print loss every 
input_dims = 224

# ============ TEXT EMBEDDINGS GRADIENT ASCENT ============
# Optimize text embeddings for cosine similarity with image embeddings
training_iterations = 300
batch_size = 12  # Adjust batch size as needed
many_tokens = 4

use_existing_embeds = True # Set 'True' to load previously computed embeddings, if available.

# ============ PGD IMAGE GENERATION (manipulation) ============
# These settings, by default, create a Deep Dream with strong adherence to the original image:

use_penultimate = False # 'True' to use penultimate layer, 'False' to use final output layer
penlayer = -2 # -2 = actual penultimate layer. -1 = final. Try -5 -> set 'use_l2 = False' below for that.

epsilon = 0.3 # Maximum deviation for projection
lr = 0.05 # Learning rate. 0.02 = low, 0.10 = high
iters = 500 # Total number of iteration steps

save_every = True # Save intermediate steps; 'False' = only save final
save_steps = 25 # Save ever n intermediate steps to 'adv_steps'

stop_gaussian_noise = 100 # when to stop adding gaussian noise to the image
use_fixed_random_seed=True # torch, numpy fixed random seed, only applies to image
range_scale = 0.00 # RGB color range restriction. 0.0: none / off. 0.15: good fit to original image with 'wiggleroom'
warmup_fraction = 0.0 # used by def 'cosine_lr_schedule'; 0.1 => 10% of total iters to warm-up

# Whether this needs adjustment heavily depends on range_scale setting (and the input image):
use_l2 = True # L2 norm correction; set "True" if you get over-bright images. 'False' if too dark.
l2_value = 1e-2 # Factor for L2 regularization

use_momentum = True # Use momentum
alpha = 0.06 # 0.02 = low, 0.10 = high

# Init image vs. Gaussian Noise / Use single image instead of tiles:
generate_single = False # 'True' to generate single image instead of 4 tiles
gaussian_init = False # 'True' to use Gaussian noise instead of image for PGD. Applies for tiles & single image alike.

# Make CLIP adhere more towards the original image while balancing with what CLIP 'saw' initially:
make_overlay = False # Overlay / inject original image with current optimization - set 'False' to turn off.
swa_start = int(0.2 * iters) # Overlay from; percentage of total iterations - ignored if make_overlay=False
swa_stop = int(0.6 * iters) # When to 'unleash' CLIP and let the AI manipulate the image w/o interference

# For post-processing bilateral filter (unrelated to CLIP, uses OpenCV):
diameter = 0 # fixed pixel neighborhood to consider; 0 = determined by sigmaSpace 
sigmaColor = 50 # 0-255, color value to consider as equal
sigmaSpace = 25 # diameter will dynamically be set based on nearby pixels


# PS: CTRL+F search for: antonym
# Have fun with the incomprehensible result =)
# ENDOFINFO





input_image_path = args.im
imagename = os.path.splitext(os.path.basename(input_image_path))[0]

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

os.makedirs('adv_PGD', exist_ok=True)
os.makedirs('adv_plots', exist_ok=True)
os.makedirs('adv_steps', exist_ok=True)
os.makedirs('longTOK', exist_ok=True)
os.makedirs('longtxtembeds', exist_ok=True)
os.makedirs('full_final', exist_ok=True)
final_folder = 'full_final'

# Function to save the original data types of model parameters and buffers
def save_original_dtypes(model):
    original_dtypes = {
        'params': {name: param.dtype for name, param in model.named_parameters()},
        'buffers': {name: buffer.dtype for name, buffer in model.named_buffers()}
    }
    return original_dtypes

# Function to restore the original data types of model parameters and buffers
def restore_original_dtypes(model, original_dtypes):
    for name, param in model.named_parameters():
        param.data = param.data.to(original_dtypes['params'][name])
    for name, buffer in model.named_buffers():
        buffer.data = buffer.data.to(original_dtypes['buffers'][name])
        
def fix_random_seed(seed: int = 6247423):
    torch.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True # Nah, waste of compute!
    #torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

model, preprocess = longclip.load(clipmodel, device=device)
original_dtypes = save_original_dtypes(model)
model = model.eval().float()


'''
 ___________  _______  ___  ___  ___________  
("     _   ")/"     "||"  \/"  |("     _   ") 
 )__/  \\__/(: ______) \   \  /  )__/  \\__/  
    \\_ /    \/    |    \\  \/      \\_ /     
    |.  |    // ___)_   /\.  \      |.  |     
    \:  |   (:      "| /  \   \     \:  |     
     \__|    \_______)|___/\___|     \__|  Transformer:
Original Code by advadnoun; X: @advadnoun
'''

prompt = longclip.tokenize('''''').numpy().tolist()[0]
prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

sideX = input_dims
sideY = input_dims

tok = simpletokenizer.SimpleTokenizer()
bests = {1000:'None', 1001:'None', 1002:'None', 1003:'None', 1004:'None'}

def clip_encode_text(gobble, text):
    x = torch.matmul(text, gobble.token_embedding.weight)  # [batch_size, n_ctx, d_model]

    x = x + gobble.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND

    x = gobble.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = gobble.ln_final(x)

    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ gobble.text_projection

    return x

def checkin(loss, tx, lll):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')

        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"longTOK/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

def load_image(img_path):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        ptt = prompt

        self.prompt = torch.zeros(batch_size, len(ptt), 49408).cuda()
        for jk, pt in enumerate(ptt):
            self.prompt[:, jk, pt] = 1 
        
        self.pad = torch.zeros(batch_size, 248 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
        return fin

lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{'params': mapper, 'lr': 5}])

nom = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
).cuda()

def augment(into):
    into = augs(into)
    return into

def ascend_txt(image):
    iii = nom(augment(image[:,:3,:,:].expand(batch_size, -1, -1, -1)))
    iii = model.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll)
    return -100*torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, batch_size).T.mean(1), tx, lll

def train(image):
    with autocast():
        loss1, tx, lll = ascend_txt(image)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll
    
def generate_target_text_embeddings(img_path, training_iterations):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image(img_path)
    
    for j in range(training_iterations):
        loss, tx, lll = train(img)
        if j % checkin_step == 0:
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {loss.mean().item()}" + Fore.RESET)
            checkin(loss, tx, lll)
   
    target_text_embedding = tx.detach()
    torch.save(target_text_embedding, f"longtxtembeds/{img_name}_text_embedding.pt")
    print(Fore.GREEN + "\nText embedding saved to 'longtxtembeds'." + Fore.RESET)
    return img, target_text_embedding, img_path
#ENDOFTEXT


'''
 ___      ___ ___  _________   
|\  \    /  /|\  \|\___   ___\ 
\ \  \  /  / | \  \|___ \  \_| 
 \ \  \/  / / \ \  \   \ \  \  
  \ \    / /   \ \  \   \ \  \ 
   \ \__/ /     \ \__\   \ \__\
    \|__|/       \|__|    \|__|  Messing with the image:
Some parts of this code originally by:
github.com/hamidkazemi22/vit-visualization
'''

pre = torch.nn.Sequential(
    RepeatBatch(12),
    ColorJitter(12, shuffle_every=False),
    GaussianNoise(12, False, 0.25, stop_gaussian_noise),
    Tile(1),  # Assuming image_size // image_size results in 1
    Jitter()
).cuda()

post = Clip().cuda()

def cosine_lr_schedule(iteration, iters, initial_lr=lr, warmup_fraction=warmup_fraction):
    warmup_iterations = int(iters * warmup_fraction)
    if iteration < warmup_iterations:
        return initial_lr * (iteration / warmup_iterations)
    else:
        return initial_lr * (1 + np.cos(np.pi * (iteration - warmup_iterations) / (iters - warmup_iterations))) / 2

def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

def scale_and_tile_image(input_image_path, input_dims, tile_folder):
    # Clear the directory before saving new tiles
    clear_directory(tile_folder)    
    imgtl = Image.open(input_image_path).convert('RGB')

    # Scale the image to 2x model input dimensions
    new_size = (2 * input_dims, 2 * input_dims)
    img_resized = imgtl.resize(new_size)    
    # Calculate the size of each tile
    tile_width = new_size[0] // 2
    tile_height = new_size[1] // 2
    
    # Cut the image into four tiles and save each
    for i in range(2):
        for j in range(2):
            left = i * tile_width
            upper = j * tile_height
            right = left + tile_width
            lower = upper + tile_height
            bbox = (left, upper, right, lower)           
            img_tile = img_resized.crop(bbox)
            
            tile_filename = f"{i+j*2+1}_{os.path.basename(input_image_path)}"            
            img_tile.save(f"{tile_folder}/{tile_filename}")

def reassemble_tiles(tile_folder: str, final_folder: str):
    os.makedirs(final_folder, exist_ok=True)
    image_files = [f for f in os.listdir(tile_folder) if f.endswith((".png", ".jpeg", ".jpg"))]

    grouped_images = {}
    for file in image_files:
        parts = file.split('_')
        tile_number = int(parts[0])
        key = '_'.join(parts[1:])
        if key not in grouped_images:
            grouped_images[key] = []
        grouped_images[key].append((tile_number, file))

    # Reassemble the image from tiles
    for key, tiles in grouped_images.items():
        tiles.sort()

        tile_images = [Image.open(os.path.join(tile_folder, tile[1])) for tile in tiles]
        tile_width, tile_height = tile_images[0].size
        combined_image = Image.new('RGB', (2 * tile_width, 2 * tile_height))

        combined_image.paste(tile_images[0], (0, 0))
        combined_image.paste(tile_images[1], (tile_width, 0))
        combined_image.paste(tile_images[2], (0, tile_height))
        combined_image.paste(tile_images[3], (tile_width, tile_height))

        combined_image.save(os.path.join(final_folder, key))

def save_intermediate_image(image_tensor, mean, std, step, image_path, tile_idx):
    image_np = image_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)   
   
    image_pil = Image.fromarray(image_np.astype(np.uint8), mode="RGB")
    image = upscale_image(image_pil, scale_factor=2)
        
    base_name = os.path.basename(image_path).split('.')[0]
    image.save(f'adv_steps/{tile_idx}_{base_name}_step{step}.png')

def upscale_image(image, scale_factor):
    width, height = image.size
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    upscaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    return upscaled_image

def evaluate_adversarial(model, image, target_text_embedding, epsilon, alpha, iters, save_every=False, save_steps=10, image_path='', use_momentum=use_momentum, generate_single=generate_single, gaussian_init=gaussian_init):
    model.eval()
    image.requires_grad = True
    clean_similarity = torch.nn.functional.cosine_similarity(model.encode_image(image), target_text_embedding).mean().item()

    if generate_single:
        # Directly optimize the entire image
        perturbed_image = pgd_attack(model, image, target_text_embedding, epsilon, alpha, iters, save_every, save_steps, image_path=image_path, use_momentum=use_momentum, make_overlay=make_overlay, lr_schedule=cosine_lr_schedule, swa_start=swa_start, gaussian_init=gaussian_init)
        reassembled_image = perturbed_image
    else:
        # Scale and tile the input image
        tile_folder = 'adv_steps'
        scale_and_tile_image(image_path, input_dims, tile_folder)

        perturbed_tiles = []
        for idx, tile_file in enumerate(sorted(os.listdir(tile_folder))):
            tile_path = os.path.join(tile_folder, tile_file)
            tile_image = load_image(tile_path).to(device).requires_grad_(True)
    
            # Pass tile index to pgd_attack and save_intermediate_image
            perturbed_tile = pgd_attack(model, tile_image, target_text_embedding, epsilon, alpha, iters, save_every, save_steps, image_path=image_path, use_momentum=use_momentum, make_overlay=make_overlay, lr_schedule=cosine_lr_schedule, swa_start=swa_start, gaussian_init=gaussian_init, tile_idx=idx + 1)
            perturbed_tiles.append(perturbed_tile)

        reassemble_tiles(tile_folder, 'full_final')
        reassembled_image_path = os.path.join('full_final', os.path.basename(image_path))
        reassembled_image = load_image(reassembled_image_path).to(device)

    # Evaluate cosine similarity on the full image vs. text embedding
    pgd_similarity = torch.nn.functional.cosine_similarity(model.encode_image(reassembled_image), target_text_embedding).mean().item()
    evaluate_all_steps(model, input_image_path, 'full_final', 'adv_plots')

    return clean_similarity, pgd_similarity, reassembled_image



   
def range_penalty(image_tensor, range_scale):
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    low_penalty = torch.clamp(-image_tensor, min=0)  # Values less than 0
    high_penalty = torch.clamp(image_tensor - 1, min=0)  # Values greater than 1
    
    # Normalize the penalty by the number of elements
    num_elements = image_tensor.numel()
    
    # Use a soft penalty function
    low_penalty = torch.sqrt(low_penalty + 1e-6)  # + to prevent sqrt(0)
    high_penalty = torch.sqrt(high_penalty + 1e-6)  # + to prevent sqrt(0)
    
    penalty = range_scale * (low_penalty.sum() + high_penalty.sum()) / num_elements
    return penalty

   
class TotalVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_wise = x[:, :, :, 1:] - x[:, :, :, :-1]
        y_wise = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(2, 3)).mean() + y_wise.norm(p=self.p, dim=(2, 3)).mean() + \
               diag_1.norm(p=self.p, dim=(2, 3)).mean() + diag_2.norm(p=self.p, dim=(2, 3)).mean()

class NormalVariation(TotalVariation):
    def forward(self, x: torch.tensor, per_sample: bool = True) -> torch.tensor:
        if per_sample:
            std = x.reshape(x.shape[0], -1).std(dim=-1).reshape(-1, 1, 1, 1)
        else:
            std = x.std()

        x = (x - x.mean()) / (std + 0.0001)
        return super(NormalVariation, self).forward(x)

class ColorVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        rolled = x.roll(shifts=1, dims=-3)
        x_wise = (x - rolled)[:, :, :, 1:] - (x - rolled)[:, :, :, :-1]
        y_wise = (x - rolled)[:, :, 1:, :] - (x - rolled)[:, :, :-1, :]
        diag_1 = (x - rolled)[:, :, 1:, 1:] - (x - rolled)[:, :, :-1, :-1]
        diag_2 = (x - rolled)[:, :, 1:, :-1] - (x - rolled)[:, :, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(2, 3)).mean() + y_wise.norm(p=self.p, dim=(2, 3)).mean() + \
               diag_1.norm(p=self.p, dim=(2, 3)).mean() + diag_2.norm(p=self.p, dim=(2, 3)).mean()

def apply_bilateral_filter_and_save(input_folder, output_suffix="_filtered", d=diameter, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace):
    image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpeg", ".jpg"))]

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Warning: Image at path {input_path} could not be read.")
            continue

        filtered_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

        base_name = os.path.splitext(image_file)[0]
        output_path = os.path.join(input_folder, f"{base_name}{output_suffix}.png")
        
        cv2.imwrite(output_path, filtered_image)
        print(f"Filtered image saved to {output_path}")

activations = {}

def hook_fn(module, input, output):
    activations['penultimate'] = output

model.visual.transformer.resblocks[penlayer].mlp.c_proj.register_forward_hook(hook_fn)

def pgd_attack(model, image, target_text_embedding, epsilon, alpha, iters, save_every=False, save_steps=10, image_path='', use_momentum=True, make_overlay=True, use_penultimate=use_penultimate, use_l2=use_l2, reg_factor_l2=l2_value, lr_schedule=cosine_lr_schedule, swa_start=swa_start, swa_stop=swa_stop, gaussian_init=gaussian_init, tile_idx=None):
    print(Fore.RED + Style.BRIGHT + f"\nGenerating PGD on image... Iterations: {iters}" + Fore.RESET)
    
    
    if gaussian_init:
        print(Fore.YELLOW + Style.BRIGHT + "Using Gaussian noise for initialization." + Fore.RESET)
        image = torch.randn_like(image).to(device).requires_grad_(True)
    
    if make_overlay:
        print(Fore.YELLOW + Style.BRIGHT + f"Injecting original image, iterations from: {swa_start} to: {swa_stop}" + Fore.RESET)
    print("\n")
    
    tv_loss_fn = TotalVariation(p=2)
    normal_var_loss_fn = NormalVariation(p=2)
    color_var_loss_fn = ColorVariation(p=2)
    
    swa_image = image.clone().detach().to(device)
    momentum = torch.zeros_like(image).to(device)
    
    for i in range(iters):
        if lr_schedule:
            alpha = lr_schedule(i, iters)

        augmented_image = pre(image)
        _ = model.encode_image(augmented_image)
        
        if use_penultimate:
            penultimate_output = activations['penultimate']
            penultimate_output = model.visual.ln_post(penultimate_output)  # Apply ln_post
            output = penultimate_output @ model.visual.proj
        else:
            output = model.encode_image(augmented_image)

        # Put a minus in front of torch, -torch.nn.functional.cosine_similarity and see what happens. :-)
        # PS: There is an antonym (many solutions, actually) to everything in CLIP (minimize cosine similarity). Usually really confusing.
        # But if you ever wanted to know what the opposite of a Tomato or a Horse might be, have fun and put a "-" here:        
        loss = torch.nn.functional.cosine_similarity(output, target_text_embedding).mean()

        if use_l2:
            l2_reg = reg_factor_l2 * torch.norm(image, p=2)
            loss -= l2_reg
        
        tv_loss = normal_var_loss_fn(image)
        loss += 0.0000001 * tv_loss

        color_var_loss = color_var_loss_fn(image)
        loss -= 0.0005 * color_var_loss

        penalty = range_penalty(image, range_scale)
        loss += penalty       
        
        model.zero_grad()
        loss.backward()

        if use_momentum:
            grad = image.grad
            if grad is not None:
                grad = grad / torch.norm(grad, p=1)
                momentum = 0.9 * momentum + grad
                image = (image + alpha * momentum.sign())
        else:
            grad = image.grad
            if grad is not None:
                image = image + alpha * grad.sign()
            else:
                raise ValueError("Gradient is None. Ensure requires_grad is set to True and backward() is called.")       

        if make_overlay and i >= swa_start and i <= swa_stop:
            weight = 0.75  # Give original == swa_image 75% weight and image 25% weight
            swa_image = (weight * swa_image + (1 - weight) * image).detach().requires_grad_(True)
            image = swa_image
                
        # Clamp perturbations to be within the epsilon range around the original image
        image = torch.clamp(image, image - epsilon, image + epsilon).detach().requires_grad_(True)
        
        # Ensure the pixel values are within the valid range [0, 1]
        image.data = post(image).data
        
        if save_every and (i + 1) % save_steps == 0:
            save_intermediate_image(image, mean, std, i + 1, image_path, tile_idx)
        if i % checkin_step == 0:
            loss_value = loss.item()
            print(f"Iteration: {i} Loss: {loss_value:.4f}")

    save_intermediate_image(image, mean, std, iters, image_path, tile_idx)
    return image





def evaluate_and_plot_similarity(model, original_image, perturbed_image, image_name, step):
    original_embedding = model.encode_image(original_image)
    perturbed_embedding = model.encode_image(perturbed_image)
    cosine_similarity = F.cosine_similarity(original_embedding, perturbed_embedding).item()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    original_np = original_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    perturbed_np = perturbed_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    original_np_uint8 = (original_np * 255).astype(np.uint8)
    perturbed_np_uint8 = (perturbed_np * 255).astype(np.uint8)

    axes[0].imshow(original_np_uint8)
    axes[0].set_title(f"Original Image")
    axes[0].axis('off')

    axes[1].imshow(perturbed_np_uint8)
    axes[1].set_title(f"Step {step}: Cosine Similarity: {cosine_similarity:.2f}")
    axes[1].axis('off')

    plt.tight_layout()
    plot_path = os.path.join('adv_plots', f"{image_name}_step{step}.png")
    plt.savefig(plot_path, pad_inches=0.1)
    plt.close()

def evaluate_all_steps(model, original_image_path, steps_folder, final_folder):
    original_image = load_image(original_image_path).to(device)
    step_images = [f for f in os.listdir(steps_folder) if f.endswith((".png", ".jpeg", ".jpg"))]
    
    for step_image in step_images:
        step_image_path = os.path.join(steps_folder, step_image)
        perturbed_image = load_image(step_image_path).to(device)
        step_number = step_image.split('_')[-1].split('.')[0][4:]
        evaluate_and_plot_similarity(model, original_image, perturbed_image, imagename, step_number)
#ENDOFVIT


image = load_image(input_image_path)

# Check whether to load existing embeddings, or compute new embeddings
if use_existing_embeds:
    embed_path = f"longtxtembeds/{imagename}_text_embedding.pt"
    if os.path.exists(embed_path):
        target_text_embedding = torch.load(embed_path).to(device)
        print(Fore.GREEN + f"\nUsing existing embedding from {embed_path}\n" + Fore.RESET)
    else:
        print(Fore.YELLOW + f"\nEmbedding not found at {embed_path}, generating new embedding." + Fore.RESET)
        print(Fore.RED + Style.BRIGHT + f"\nGenerating Text Embeddings for Image... Iterations: {training_iterations}. CLIP's opinion:\n" + Fore.RESET)
        image, target_text_embedding, image_path = generate_target_text_embeddings(input_image_path, training_iterations)
else:
    print(Fore.RED + Style.BRIGHT + f"\nGenerating Text Embeddings for Image... Iterations: {training_iterations}. CLIP's opinion:\n" + Fore.RESET)
    image, target_text_embedding, image_path = generate_target_text_embeddings(input_image_path, training_iterations)

torch.cuda.empty_cache()
restore_original_dtypes(model, original_dtypes)
model = model.eval()

if use_fixed_random_seed:
    fix_random_seed(seed=6247423)
    print(Fore.BLUE + Style.BRIGHT + f"Using fixed random seed." + Fore.RESET)

# Use Projected Gradient Descent (PGD) to introduce strong perturbations and "dream" towards the text embeddings
clean_similarity, pgd_similarity, reassembled_image = evaluate_adversarial(model, image, target_text_embedding, epsilon, alpha, iters, save_every, save_steps, image_path=input_image_path)


# Plot & save muchly
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
image_np = image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
image_np = (image_np * 255).astype(np.uint8)
reassembled_np = reassembled_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
reassembled_np = (reassembled_np * 255).astype(np.uint8)

axes[0].imshow(image_np)
axes[0].set_title(f"Original Similarity: {clean_similarity:.2f}")
axes[0].axis('off')

axes[1].imshow(reassembled_np)
axes[1].set_title(f"PGD Similarity: {pgd_similarity:.2f}")
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f"adv_plots/adv_{imagename}_e{epsilon}-a{alpha}-i{iters}.png", pad_inches=0.1)
plt.close()

reassembled_image_pil = Image.fromarray(reassembled_np.astype(np.uint8), mode="RGB")
reassembled_image_pil = upscale_image(reassembled_image_pil, scale_factor=2)
reassembled_image_pil.save(os.path.join('full_final', f'{imagename}.png'))

full_final_folder = 'full_final'
apply_bilateral_filter_and_save(full_final_folder)
print(Fore.GREEN + Style.BRIGHT + f"\nAll done. Check the output folders!\n")