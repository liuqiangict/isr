import torch
import torch.nn as nn
import models.ops as ops
import torch.nn.functional as F


__all__ = [ "drln" ]


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 3, 1, 3, 3)
        self.c2 = ops.BasicBlock(channel , channel // reduction, 3, 1, 5, 5)
        self.c3 = ops.BasicBlock(channel , channel // reduction, 3, 1, 7, 7)
        self.c4 = ops.BasicBlockSig((channel // reduction)*3, channel , 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.ResidualBlock(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels * 2, out_channels * 2)
        self.r3 = ops.ResidualBlock(in_channels * 4, out_channels * 4)
        self.g = ops.BasicBlock(in_channels * 8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 =  x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)
                
        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)
               
        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out

class DRLM(nn.Module):
    def __init__(self, chs, scale):
        super(DRLM, self).__init__()
        self.b = Block(chs, chs)
        self.c = ops.BasicBlock(chs * scale, chs, 3, 1, 1)


    def forward(self, x):
        c, o = x
        b = self.b(o)
        c = torch.cat([c, b], dim=1)
        o = self.c(c)

        return (c, o)


class DenseBlock_3(nn.Module):
    def __init__(self, chs):
        super(DenseBlock_3, self).__init__()
        body = [DRLM(chs, i) for i in range(2, 5)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        c, o = x
        out = self.body(x)
        c_out, o_out = out
        a = o_out + c
        return (a, o_out)


class DenseBlock_4(nn.Module):
    def __init__(self, chs):
        super(DenseBlock_4, self).__init__()
        body = [DRLM(chs, i) for i in range(2, 6)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        c, o = x
        out = self.body(x)
        c_out, o_out = out
        a = o_out + c
        return (a, o_out)

class DRLN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.scale = 2
        chs = 64

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.head = nn.Conv2d(3, chs, 3, 1, 1)

        n_resblocks = 6

        self.n_resblocks_3 = n_resblocks //  3 * 2
        self.n_resblocks_4 = n_resblocks - self.n_resblocks_3

        body_1 = [DenseBlock_3(chs) for _ in range(self.n_resblocks_3)]
        body_2 = [DenseBlock_4(chs) for _ in range(self.n_resblocks_4)]

        self.body_1 = nn.Sequential(*body_1)
        self.body_2 = nn.Sequential(*body_2)

        self.upsample = ops.UpsampleBlock(chs, self.scale , multi_scale=False)
        self.tail = nn.Conv2d(chs, 3, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        c = o = x

        out = (c, o)
        out = self.body_1(out)
        out = self.body_2(out)
        a, o = out

        b_out = a + x
        out = self.upsample(b_out, scale=self.scale )

        out = self.tail(out)
        f_out = self.add_mean(out)

        return f_out


def _drln(arch):
    model = DRLN()
    return model

def drln(args):
    return _drln('DRLN')

