import os
import sys

import math
import numpy
import cv2
import imageio
import lpips

import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    shave = scale
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return torch.tensor([-10 * math.log10(mse)], device=sr.device)

def calc_ssim(img1, img2):
    def ssim(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(numpy.float64)
        img2 = img2.astype(numpy.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = numpy.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    device = img1.device

    img1 = img1[0].permute(1, 2, 0).cpu().numpy()
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()
    border = 0
    img1_y = numpy.dot(img1, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    img2_y = numpy.dot(img2, [65.738, 129.057, 25.064]) / 256.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    if img1_y.ndim == 2:
        return torch.tensor([ssim(img1_y, img2_y)], device=device)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return torch.tensor([numpy.array(ssims).mean()], device=device)
        elif img1.shape[2] == 1:
            return torch.tensor([ssim(numpy.squeeze(img1), numpy.squeeze(img2))], device=device)
    else:
        raise ValueError('Wrong input image dimensions.')

def calc_lpips(loss_fn, img1, img2):
    factor = 255. / 2.
    cent = 1.
    img1 = img1 / factor - cent
    img2 = img2 / factor - cent

    # Compute distance
    with torch.no_grad():
        dist = loss_fn.forward(img1, img2).squeeze().unsqueeze(dim=0)
    return dist

def save_results(data, output_folder, filename, scale):
    #filename = os.path.join(output_folder, '{}_x{}'.format(filename[0], scale))
    filename = os.path.join(output_folder, filename[0])
    b, c, h, w = data.shape
    img = data.reshape(c, h, w).round().clamp_(0., 255.).byte().permute(1, 2, 0).cpu().numpy()
    imageio.imsave('{}.png'.format(filename), img)

