
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class VGG(nn.Module):
    def __init__(self, conv_index='54', rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        for k, v in self.vgg.named_parameters():
            v.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        vgg_hr = _forward(hr)

        loss = F.l1_loss(vgg_sr, vgg_hr)
        #loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
'''

class VGG(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True):
        super(VGG, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = models.vgg19_bn(pretrained=True)
        else:
            model = models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, sr, hr):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            sr = (sr - self.mean) / self.std
            hr = (hr - self.mean) / self.std

        vgg_sr = self.features(sr)
        vgg_hr = self.features(hr)

        loss = F.l1_loss(vgg_sr, vgg_hr)

        return loss
'''
