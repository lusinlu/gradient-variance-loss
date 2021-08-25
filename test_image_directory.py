# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from vdsr import VDSR

parser = argparse.ArgumentParser(description="Gradient Variance loss for structure-enhanced super-resolution")
parser.add_argument("--dataroot", type=str,help="The directory path where the image needs ")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 3, 4, 8],
                    help="Image scaling ratio. (default: 4).")
parser.add_argument("--weights", type=str, help="path to the model weights")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")

args = parser.parse_args()
print(args)

try:
    os.makedirs("result")
except OSError:
    pass

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: CUDA device detected, you can run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
model = VDSR().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))


dataroot = args.dataroot
scale_factor = args.scale_factor

for filename in os.listdir(dataroot):
    # Open image
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

    # before converting the result in RGB
    out_image.save(f"result/{filename}")

