import os
import glob
import torch
import numpy as np
import math
import imageio

def load_filenames(datafolder, ground_truth=''):
    assert os.path.exists(datafolder), f'{datafolder} does not exist'
    lr_images = glob.glob(os.path.join(datafolder, '*' + '.png'))

    # Only load ground truth if specified. hr image name must be same as lr image
    hr_images = []
    if ground_truth:
        for lr_i in lr_images:
            hr_images.append(os.path.join(ground_truth, os.path.basename(lr_i)))

    return lr_images, hr_images

def image_to_tensor(image_path, cpu=False):
    image = imageio.imread(image_path)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    
    if image.shape[2] == 1:
        image = np.concatenate([image] * 3, 2)
    
    image = np.ascontiguousarray(image.transpose((2, 0, 1)))
    image = torch.from_numpy(image).float()

    if not cpu:
        image = image.cuda()

    return image

def y_channel(image):
    Kr = np.float64(0.299)
    Kb = np.float64(0.114)

    image = image.to(dtype=torch.float64) / 255
    gray_coeffs = image.new_tensor([Kr, 1-Kr-Kb, Kb]).view(1, 3, 1, 1)
    
    image = image.mul(gray_coeffs).sum(dim=1, keepdim=True)*np.float64(219)/255 + np.float64(16)/255
    
    assert image.size()[1] == 1
    return image * 255

def calculate_psnr(sr_image, hr_image, shave=0):
    diff_image = sr_image - hr_image
    diff_image = diff_image[..., shave:-shave, shave:-shave]
    mse = diff_image.pow(2).mean()

    return 10 * math.log10(255**2 / mse)

def save_images(sr_images, filenames, folder='output_images'):
    for i, image in enumerate(sr_images):
        save_path = os.path.join(folder, os.path.basename(filenames[i]))
        image = image.squeeze().byte().permute(1,2,0).cpu()
        imageio.imwrite(save_path, image.numpy())
