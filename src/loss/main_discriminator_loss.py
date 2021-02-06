import os
import sys
import datetime
import imageio 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        in_channels = 3 #args.n_colors
        out_channels = 64
        depth = 7
        self.patch_size = 320

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))

        #patch_size = args.patch_size // (2**((depth + 1) // 2))
        patch_size = self.patch_size // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output


class dis_model(nn.Module):
    def __init__(self):
        super(dis_model, self).__init__()
        print('Build Model.')
        self.dis = Discriminator()
        
        print('Load checkpoint.')
        self.pretrained = '/relevance-nfs/users/qiangliu/Models/turingISR/v1/base/train_div2k_flickr_isr_v1_base_scale_4_noise_vgg_rgan_1vs100vs05_var_15/checkpoint_10000'
        model_path = os.path.join(self.pretrained, 'GAN_D_pytorch_model.bin')
        checkpoint = torch.load(model_path, 'cpu')
        self.dis.load_state_dict(checkpoint, strict=True)

        self.dis = self.dis.cuda()

    def forward(self, img):
        output = self.dis(img)
        return output


if __name__ == "__main__":
    m = dis_model()

    source_folder = 'dis_test'
    files = os.listdir(source_folder)
    for f in files:
        print(datetime.datetime.now(), f)
        img = imageio.imread(os.path.join(source_folder, f))
        h, w, c = img.shape
        s_h = (h - 320) // 2
        s_w = (w - 320) // 2
        img = img[s_h:s_h + 320, s_w:s_w + 320, :]

        lr = torch.tensor([img], dtype=torch.float).cuda().permute(0, 3, 1, 2)
        res = m(lr)
        print(res)

