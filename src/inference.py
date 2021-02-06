import argparse
import json
import os
import random
import shutil
import time
import math
import logging
import warnings
import numpy as np
import tempfile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import transforms
import datasets
import models
from apex import amp
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm, trange

import deepspeed

from utils import calc_psnr
from utils import calc_ssim
from utils import save_results

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class trainer:
    def __init__(self, args):
        self.args = args
        self.summ_writer = SummaryWriter(args.output_dir)
        self.global_step = 0

        self.model = self.build_model()
        self.valid_set, self.valid_loader = self.build_dataset()

    def build_dataset(self):
        data_transform = transforms.Compose([
            transforms.ToTorchTensor(),
        ])

        if self.args.inference:
            valid_set = datasets.HRDatasetFolder(self.args.eval_datadir, self.args.scale, patch_size=-1, transform=data_transform, is_same_size=self.args.is_same_size)
        elif self.args.arch in ['transformer', 'longformer']:
            valid_set = datasets.HRDatasetFolder(self.args.eval_datadir, self.args.scale, patch_size=self.args.patch_size, transform=data_transform, is_same_size=self.args.is_same_size)
        else:
            valid_set = datasets.HRDatasetFolder(self.args.eval_datadir, self.args.scale, transform=data_transform, is_same_size=self.args.is_same_size)

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=1, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)

        return valid_set, valid_loader


    def build_model(self):
        logger.info("Creating model '{}'".format(self.args.arch))
        model = models.__dict__[self.args.arch](self.args)

        if self.args.distributed:
            torch.cuda.set_device(self.args.gpu)
            model.cuda(self.args.gpu)
            dist.init_process_group(backend=self.args.dist_backend)
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu)
            model = model.cuda(self.args.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()

        if self.args.pretrained:
            model_path = os.path.join(self.args.pretrained, 'pytorch_model.bin')
            if os.path.isfile(model_path):
                logger.info("Loading model from {}".format(model_path))
                checkpoint = torch.load(model_path, 'cpu')
                model.load_state_dict(checkpoint, strict=True)

        return model

    def is_local_master(self):
        return self.args.local_rank in [-1, 0]
    
    def is_global_master(self):
        return self.args.distributed == False or dist.get_rank() == 0

    def validate(self):
        # switch to evaluate mode
        self.model.eval()
        psnrs = None
        ssims = None
        output_dir = os.path.join(self.args.output_dir, 'validate_' + str(self.global_step))
        if self.is_global_master():
            os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for hr, lr, file_name in tqdm(self.valid_loader, disable=(not self.is_local_master())):
                hr = hr.cuda(self.args.gpu, non_blocking=True)
                lr = lr.cuda(self.args.gpu, non_blocking=True)
                sr = self.model(lr)

                psnr = calc_psnr(hr, sr, self.args.scale, self.args.rgb_range)
                ssim = calc_ssim(hr, sr)
                save_results(sr, output_dir, file_name, self.args.scale)

                if psnrs is None:
                    psnrs = psnr.detach()
                else:
                    psnrs = torch.cat((psnrs, psnr.detach()), dim=0)
                if ssims is None:
                    ssims = ssim.detach()
                else:
                    ssims = torch.cat((ssims, ssim.detach()), dim=0)


        if self.is_global_master():
            logger.info('Global Step: {}, PSNR: {}, SSIM: {}.'.format(self.global_step, 
                    torch.mean(psnrs).cpu().numpy(), 
                    torch.mean(ssims).cpu().numpy()))
        self.summ_writer.add_scalar('Eval/psnr', torch.mean(psnrs).cpu().numpy(), self.global_step)
        self.summ_writer.add_scalar('Eval/ssim', torch.mean(ssims).cpu().numpy(), self.global_step)

    # Only giving low resolution image, to inference super resolution images
    def inference(self):
        self.model.eval()
        output_dir = os.path.join(self.args.output_dir, 'Inference_Scale_' + str(self.args.scale))
        if self.is_global_master():
            os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for lr, file_name in tqdm(self.valid_loader, disable=(not self.is_local_master())):
                lr = lr.cuda(self.args.gpu, non_blocking=True)
                sr = self.model(lr)
                save_results(sr, output_dir, file_name, self.args.scale)

    def main_worker(self):
        if self.args.gpu is not None:
            logger.info("Use GPU: {} for training.".format(self.args.gpu))

        cudnn.benchmark = True

        if self.args.inference:
            self.inference()
            return

        if self.args.evaluate:
            self.validate()
            return

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--eval_datadir', metavar='DIR', help='path to dataset')
    parser.add_argument('--output_dir', default = './', type=str, help='path to save model output.')

    parser.add_argument('--arch', metavar='ARCH', default='drln', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--pretrained', default=None, type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model_name_or_path', default=None, type=str,
                        help='model path for transformer based model.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--inference', dest='inference', action='store_true',
                        help='Do inference on low resolution images')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--fp16_level', default="O0", type=str,
                        help='FP16 optimization level')
    
    parser.add_argument('--scale', type=int, default=2,
                        help='super resolution scale')
    parser.add_argument('--patch_size', type=int, default=192,
                        help='output patch size')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    parser.add_argument('--is_same_size', action='store_true',
                        help='Using same size of the image.')

    parser.add_argument('--n_resblocks', type=int, default=16,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--dilation', action='store_true',
                        help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')

    # Option for Residual channel attention network (RCAN)
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=str, default='200',
                        help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--gclip', type=float, default=0,
                        help='gradient clipping threshold (0 = no clipping)')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, 
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--loss', type=str, default='1*L1',
                        help='loss function configuration')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.world_size == -1 and args.local_rank != -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    if args.local_rank != -1:
        args.gpu = args.local_rank

    args.distributed = args.world_size > 1

    trainer(args).main_worker()

if __name__ == '__main__':
    main()
