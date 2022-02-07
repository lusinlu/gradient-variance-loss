
import argparse
import math
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
from vdsr import VDSR
from gradient_variance_loss import GradientVariance
from torch.utils.tensorboard import SummaryWriter
from datasets import TrainingDataset, ValidationDataset
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="VDSR training with the GradientVariance loss")
parser.add_argument("--dataroot", type=str, help="Path to datasets")
parser.add_argument("--epochs", default=30, type=int, metavar="N",
                    help="Number of total epochs to run")
parser.add_argument("--image-size", type=int, default=128,
                    help="Size of the data crop (squared assumed)")
parser.add_argument("-b", "--batch-size", default=64, type=int,
                    metavar="N", help="mini-batch size.")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--scale-factor", type=int, default=2, choices=[2, 3, 4],
                    help="super-resolution scaling factor")
parser.add_argument("--weights",help="Path to pre-trained weights")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--loss_patch_size", type=int, default=8,
                    help="patch size for variance calculations")
parser.add_argument("--gradloss_weight", type=float, default=0.01,
                    help="weights of the gradient variance loss")
parser.add_argument("--clip", type=float, default=0.4, help="clipping gradients")


args = parser.parse_args()
print(args)

try:
    os.makedirs("weights")
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: CUDA device available, consider run with --cuda")

train_dataset = TrainingDataset(f"{args.dataroot}/train",
                                  image_size=args.image_size,
                                  scale_factor=args.scale_factor)
val_dataset = ValidationDataset(f"{args.dataroot}/val",
                                scale_factor=args.scale_factor)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=4)

device = torch.device("cuda:0" if args.cuda else "cpu")

model = VDSR().to(device)

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device))

criterion = nn.MSELoss().to(device)
grad_criterion = GradientVariance(patch_size=args.loss_patch_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 10, 20], gamma=0.1)

best_psnr = 0.
summary = SummaryWriter()

for epoch in range(args.epochs):
    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for iteration, (inputs, target) in progress_bar:
        optimizer.zero_grad()

        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)

        loss_mse =  criterion(output, target)
        loss_grad = args.gradloss_weight * grad_criterion(output, target)
        loss = loss_mse + loss_grad

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),args.clip)

        optimizer.step()


        progress_bar.set_description(f"[{epoch + 1}/{args.epochs}][{iteration + 1}/{len(train_dataloader)}] "
                                     f"Loss mse: {loss_mse.item():.6f} " f"Loss grad: {loss_grad.item():.6f} ")
    summary.add_scalar('Loss/train', loss.detach().cpu().numpy(), epoch)

    # Test
    model.eval()
    avg_psnr = 0.
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for iteration, (inputs, target) in progress_bar:
            inputs, target = inputs.to(device), target.to(device)

            prediction = model(inputs)
            mse = criterion(prediction, target)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
            progress_bar.set_description(f"Epoch: {epoch + 1} [{iteration + 1}/{len(val_dataloader)}] "
                                         f"Loss mse: {loss_mse.item():.6f} "
                                         f"Loss grad: {loss_grad.item():.6f} "
                                         f"PSNR: {psnr:.2f}.")

    print(f"Average PSNR: {avg_psnr / len(val_dataloader):.2f} dB.")
    summary.add_scalar('psnr/val',avg_psnr / len(val_dataloader), epoch)

    scheduler.step()

    # Save model
    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"weights/vdsr_{args.scale_factor}x_epoch_{epoch + 1}.pth")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), f"weights/vdsr_{args.scale_factor}x.pth")
