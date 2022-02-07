import argparse
import math
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
from vdsr import VDSR
from datasets import ValidationDataset
from utils import calc_ssim

parser = argparse.ArgumentParser(description="VDSR model benchmark ")
parser.add_argument("--dataroot", type=str, help="Path to datasets")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers")
parser.add_argument("--scale-factor", type=int, required=True, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor.")
parser.add_argument("--weights", type=str, required=True,
                    help="Path to weights.")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")

args = parser.parse_args()
print(args)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: CUDA device available, consider run with --cuda")

dataset = ValidationDataset(f"{args.dataroot}/val",
                            scale_factor=args.scale_factor)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

# define model and load pretrained weights
model = VDSR().to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))
criterion = nn.MSELoss().to(device)

model.eval()
avg_psnr = 0.
avg_mse = 0.
avg_ssim = 0.
with torch.no_grad():
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for iteration, (inputs, target) in progress_bar:
        inputs, target = inputs.to(device), target.to(device)

        prediction = model(inputs)
        mse = criterion(prediction, target)
        avg_mse += mse
        psnr = 10 * math.log10(1 / mse.item())

        avg_psnr += psnr
        ssim = calc_ssim(prediction * 255., target * 255., rgb_range=1, scale=2, dataset='dataset')
        avg_ssim += ssim

        progress_bar.set_description(f"[{iteration + 1}/{len(dataloader)}] "
                                     f"SSIM: {ssim:.6f} PSNR: {psnr:.6f}.")

    print(f"Average PSNR: {avg_psnr / len(dataloader):.2f} dB.")
    print(f"Average SSIM: {avg_ssim / len(dataloader):.6f}")

