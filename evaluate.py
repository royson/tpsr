import utils
import argparse
import torch
from model import TPSR
from torch.utils.data import dataloader

parser = argparse.ArgumentParser(description='Evaluation Code for TPSR Evaluation')
parser.add_argument('--model', type=str, default='',
                    help='pre-trained model path')
parser.add_argument('--input_images', type=str, default='input_images',
                    help='input images folder. Default: Set5 images.')
parser.add_argument('--output_images', type=str, default='output_images',
                    help='upsampled images folder')
parser.add_argument('--ground_truth', type=str, default='ground_truth',
                    help='ground_truth folder for PSNR evaluation')
parser.add_argument('--cpu', action='store_true',
                    help='use CPU')
args = parser.parse_args()

scale = 4
### Data Loading
lr, hr = utils.load_filenames(args.input_images, args.ground_truth)

lr_images = [utils.image_to_tensor(l, cpu=args.cpu) for l in lr]
hr_images = [utils.image_to_tensor(h, cpu=args.cpu) for h in hr]

### Model
m = TPSR(scale=scale)
if not args.cpu:
    m.cuda()

kwargs = {}
if args.cpu:
    kwargs = {'map_location': lambda storage, loc: storage}
if args.model:
    m.load_state_dict(torch.load(args.model, **kwargs), strict=False)

torch.set_grad_enabled(False)

sr_images = []
psnr = []

### Upscale
for i, l in enumerate(lr_images):
    sr_image = m(l.unsqueeze(0)).clamp(0, 255).round()
    if len(hr_images) > 0:
        sr_image_y = utils.y_channel(sr_image)
        psnr.append(utils.calculate_psnr(sr_image_y, utils.y_channel(hr_images[i]), shave=scale))
    sr_images.append(sr_image)

if len(psnr) > 0:
    print(f"Average PSNR: {sum(psnr) / len(psnr)}")

### Save
print(f"Saving output images to {args.output_images}.")
utils.save_images(sr_images, filenames=lr, folder=args.output_images)