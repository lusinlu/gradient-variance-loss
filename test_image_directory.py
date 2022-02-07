import argparse
import os

import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from vdsr import VDSR

parser = argparse.ArgumentParser(description="Gradient Variance loss for structure-enhanced super-resolution")
parser.add_argument("--dataroot", type=str, required=True, help="The directory path where the image needs ")
parser.add_argument("--scale-factor", type=int, default=2, choices=[2, 3, 4, 8],
                    help="Image scaling ratio. (default: 2).")
parser.add_argument("--weights", type=str, required=True, help="path to the model weights")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")

args = parser.parse_args()
print(args)

try:
    os.makedirs("result")
except OSError:
    pass

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: CUDA device available, consider run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# define model
model = VDSR().to(device)

# Load pretrained weights
model.load_state_dict(torch.load(args.weights, map_location=device))


dataroot = args.dataroot
scale_factor = args.scale_factor

for filename in os.listdir(dataroot):
    # open image and upscale
    image = Image.open(f"{dataroot}/{filename}")
    image_width = int(image.size[0] * scale_factor)
    image_height = int(image.size[1] * scale_factor)
    image = image.resize((image_width, image_height), Image.BICUBIC)

    preprocess = transforms.ToTensor()
    inputs = preprocess(image).view(1, -1, image.size[1], image.size[0])

    inputs = inputs.to(device)

    out = model(inputs)
    out = out.cpu()
    out_image = out[0].detach().numpy()
    out_image *= 255.0
    out_image = out_image.clip(0, 255).transpose(1, 2, 0)
    out_image = Image.fromarray(np.uint8(out_image))

    out_image.save(f"result/{filename}")

