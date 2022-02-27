from torch import nn
import torch
import torch.nn.functional as F


class GradientVariance(nn.Module):
    """Class for calculating GV loss between to RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
        """
    def __init__(self, patch_size, cpu=False):
        super(GradientVariance, self).__init__()
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()
        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

    def forward(self, output, target):
        # converting RGB image to grayscale
        gray_output = 0.2989 * output[:, 0:1, :, :] + 0.5870 * output[:, 1:2, :, :] + 0.1140 * output[:, 2:, :, :]
        gray_target = 0.2989 * target[:, 0:1, :, :] + 0.5870 * target[:, 1:2, :, :] + 0.1140 * target[:, 2:, :, :]

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(gray_target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(gray_target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(gray_output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(gray_output, self.kernel_y, stride=1, padding=1)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient maps
        gradvar_loss = F.mse_loss(var_target_x, var_output_x) + F.mse_loss(var_target_y, var_output_y)

        return gradvar_loss


