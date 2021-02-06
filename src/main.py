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

from loss.vgg import VGG
from loss.adversarial import Adversarial
import lpips

from utils import AverageMeter
from utils import calc_psnr
from utils import calc_ssim
from utils import calc_lpips
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
        self.optimizer = self.build_optimizer(self.args, self.model)
        self.losses = self.build_loss()
        self.train_set, self.valid_set, self.train_loader, self.valid_loader = self.build_dataset()
        self.lpips_loss_fn = lpips.LPIPS(net='alex',version='0.1').cuda()
        self.lpips_loss_fn.eval()
        self.enable_GAN = 'GAN' in self.args.loss 

    def build_dataset(self):
        data_transform = transforms.Compose([
            transforms.ToTorchTensor(),
        ])

        train_set = datasets.HRDatasetFolder(
            self.args.train_datadir,
            self.args.scale,
            patch_size=self.args.patch_size,
            transform=data_transform,
            image_extension=self.args.train_image_ext,
            is_same_size=self.args.is_same_size,
            lr_root=self.args.train_datadir_lr,
            noise_root=self.args.noise_datadir, 
            noise_ratio=self.args.noise_ratio,
            quality_range=self.args.quality_range,
            blur_types=self.args.blur_types)

        if self.args.inference:
            valid_set = datasets.HRDatasetFolder(
                self.args.eval_datadir,
                self.args.scale,
                patch_size=-1,
                transform=data_transform,
                image_extension=self.args.eval_image_ext,
                is_same_size=self.args.is_same_size,
                lr_root=self.args.eval_datadir_lr)
        else:
            valid_set = datasets.HRDatasetFolder(
                self.args.eval_datadir,
                self.args.scale,
                transform=data_transform,
                image_extension=self.args.eval_image_ext,
                is_same_size=self.args.is_same_size,
                lr_root=self.args.eval_datadir_lr)

        if self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            #valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set)
        else:
            train_sampler = None
            #valid_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.args.batch_size, shuffle=(train_sampler is None),
            num_workers=self.args.workers, pin_memory=True, sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=1, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)

        return train_set, valid_set, train_loader, valid_loader


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

    def build_optimizer(self, args, model):
        # optimizer
        trainable = filter(lambda x: x.requires_grad, model.parameters())
        kwargs_optimizer = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}

        if args.optimizer == 'SGD':
            optimizer_class = optim.SGD
            kwargs_optimizer['momentum'] = args.momentum
        elif args.optimizer == 'ADAM':
            optimizer_class = optim.Adam
            kwargs_optimizer['betas'] = args.betas
            kwargs_optimizer['eps'] = args.epsilon
        elif args.optimizer == 'RMSprop':
            optimizer_class = optim.RMSprop
            kwargs_optimizer['eps'] = args.epsilon

        # scheduler
        milestones = list(map(lambda x: int(x), args.decay.split('-')))
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
        scheduler_class = lr_scheduler.MultiStepLR

        class CustomOptimizer(optimizer_class):
            def __init__(self, *args, **kwargs):
                super(CustomOptimizer, self).__init__(*args, **kwargs)

            def _register_scheduler(self, scheduler_class, **kwargs):
                self.scheduler = scheduler_class(self, **kwargs)

            def save(self, save_dir):
                torch.save(self.state_dict(), self.get_dir(save_dir))

            def load(self, load_dir, epoch=1):
                self.load_state_dict(torch.load(self.get_dir(load_dir)))
                if epoch > 1:
                    for _ in range(epoch): self.scheduler.step()

            def get_dir(self, dir_path):
                return os.path.join(dir_path, 'optimizer.pt')

            def schedule(self):
                self.scheduler.step()

            def get_lr(self):
                return self.scheduler.get_lr()[0]

            def get_last_epoch(self):
                return self.scheduler.last_epoch
    
        optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
        optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)

        '''
        if args.pretrained:
            optimizer_path = os.path.join(args.pretrained, 'optimizer.bin')
            if os.path.isfile(optimizer_path):
                logger.info("Loading optimizer from {}".format(optimizer_path))
                checkpoint = torch.load(optimizer_path, 'cpu')
                optimizer.load_state_dict(checkpoint)
        '''

        return optimizer

    def build_loss(self):
        losses = []
        for loss in self.args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'vgg':
                loss_function = VGG(rgb_range=self.args.rgb_range).cuda()
            elif 'GAN' in loss_type:
                loss_function = Adversarial(self.args, gan_type=loss_type, build_optimizer=self.build_optimizer).cuda()
                self.gan_dis_optimizer = loss_function.optimizer
                self.gan_dis_model = loss_function.dis
            losses.append({'type': loss_type, 'weight': float(weight), 'function': loss_function})

        return losses

    def distributed_concat(self, tensor):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)
        return concat
    
    def validate(self):
        # switch to evaluate mode
        self.model.eval()
        psnrs = None
        ssims = None
        lpipses = None
        output_dir = os.path.join(self.args.output_dir, 'validate_' + str(self.global_step))
        if self.is_global_master():
            os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for hr, lr, file_name in tqdm(self.valid_loader, disable=(not self.is_local_master())):
                hr = hr.cuda(self.args.gpu, non_blocking=True)
                lr = lr.cuda(self.args.gpu, non_blocking=True)
                if self.args.fp16_level == "O2":
                    lr = lr.half()
                    hr = hr.half()
                if self.args.arch in ['transformer', 'longformer', 'sparse']: # have to do patch inference here, concate all patches together
                    sr = torch.tensor([], dtype=lr.dtype, device=self.args.gpu)
                    b, c, h, l = lr.shape
                    ch_range = h // (self.args.patch_size - 2 * self.args.edge_size)
                    if h % (self.args.patch_size - 2 * self.args.edge_size) != 0:
                        ch_range += 1
                    cl_range = l // (self.args.patch_size - 2 * self.args.edge_size)
                    if l % (self.args.patch_size - 2 * self.args.edge_size) != 0:
                        cl_range += 1
                    for ch in range(ch_range):
                        tmp_sr = torch.tensor([], dtype=lr.dtype, device=self.args.gpu)
                        if ch == 0:
                            h_st = 0
                            h_en = self.args.patch_size 
                        elif ch == ch_range - 1:
                            h_st = h - self.args.patch_size
                            h_en = h
                        else:
                            h_st = ch * (self.args.patch_size - 2 * self.args.edge_size) - self.args.edge_size
                            h_en = h_st + self.args.patch_size
                        for cl in range(cl_range):
                            if cl == 0:
                                l_st = 0
                                l_en = self.args.patch_size
                            elif cl == cl_range - 1:
                                l_st = l - self.args.patch_size
                                l_en = l
                            else:
                                l_st = cl * (self.args.patch_size - 2 * self.args.edge_size) - self.args.edge_size
                                l_en = l_st + self.args.patch_size
                            c_lr = lr[:, :, h_st:h_en, l_st:l_en]

                            c_sr = self.model(c_lr)
                            if cl == 0:
                                c_sr = c_sr[:, :, :, :(self.args.patch_size - 2 * self.args.edge_size)]
                            elif cl == cl_range - 1:
                                c_sr = c_sr[:, :, :, (cl * (self.args.patch_size - 2 * self.args.edge_size) - l):]
                            elif self.args.edge_size != 0:
                                c_sr = c_sr[:, :, :, self.args.edge_size:-self.args.edge_size]

                            tmp_sr = torch.cat((tmp_sr, c_sr), dim=3)
                        if ch == 0:
                            tmp_sr = tmp_sr[:, :, :(self.args.patch_size - 2 * self.args.edge_size), :]
                        elif ch == ch_range - 1:
                            tmp_sr = tmp_sr[:, :, (ch * (self.args.patch_size - 2 * self.args.edge_size) - h):, :]
                        elif self.args.edge_size != 0:
                            tmp_sr = tmp_sr[:, :, self.args.edge_size:-self.args.edge_size, :]

                        sr = torch.cat((sr, tmp_sr), dim=2)
                else:
                    sr = self.model(lr)

                psnr = calc_psnr(hr, sr, self.args.scale, self.args.rgb_range)
                ssim = calc_ssim(hr, sr)
                lpips = calc_lpips(self.lpips_loss_fn, hr, sr)
                save_results(sr, output_dir, file_name, self.args.scale)
                logger.info('Image: {}, PSNR: {}, SSIM: {}, LPIPS: {}.'.format(file_name, psnr, ssim, lpips))

                if psnrs is None:
                    psnrs = psnr.detach()
                else:
                    psnrs = torch.cat((psnrs, psnr.detach()), dim=0)
                if ssims is None:
                    ssims = ssim.detach()
                else:
                    ssims = torch.cat((ssims, ssim.detach()), dim=0)
                if lpipses is None:
                    lpipses = lpips.detach()
                else:
                    lpipses = torch.cat((lpipses, lpips.detach()), dim=0)

        #psnrs = self.distributed_concat(psnrs)
        #ssims = self.distributed_concat(ssims)

        if self.is_global_master():
            logger.info('Global Step: {}, PSNR: {}, SSIM: {}, LPIPS: {}.'.format(self.global_step,
                    torch.mean(psnrs).cpu().numpy(), 
                    torch.mean(ssims).cpu().numpy(),
                    torch.mean(lpipses).cpu().numpy()
                    ))
        self.summ_writer.add_scalar('Eval/psnr', torch.mean(psnrs).cpu().numpy(), self.global_step)
        self.summ_writer.add_scalar('Eval/ssim', torch.mean(ssims).cpu().numpy(), self.global_step)
        self.summ_writer.add_scalar('Eval/lpips', torch.mean(lpipses).cpu().numpy(), self.global_step)

    # Only giving low resolution image, to inference super resolution images
    def inference(self):
        self.model.eval()
        output_dir = os.path.join(self.args.output_dir, 'Inference_Scale_' + str(self.args.scale))
        if self.is_global_master():
            os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for lr, file_name in tqdm(self.valid_loader, disable=(not self.is_local_master())):
                lr = lr.cuda(self.args.gpu, non_blocking=True)
                if self.args.fp16_level == "O2":
                    lr = lr.half()
                if self.args.arch in ['transformer', 'longformer', 'sparse']: # have to do patch inference here, concate all patches together
                    sr = torch.tensor([], dtype=lr.dtype, device=self.args.gpu)
                    b, c, h, l = lr.shape
                    ch_range = h // (self.args.patch_size - 2 * self.args.edge_size)
                    if h % (self.args.patch_size - 2 * self.args.edge_size) != 0:
                        ch_range += 1
                    cl_range = l // (self.args.patch_size - 2 * self.args.edge_size)
                    if l % (self.args.patch_size - 2 * self.args.edge_size) != 0:
                        cl_range += 1
                    for ch in range(ch_range):
                        tmp_sr = torch.tensor([], dtype=lr.dtype, device=self.args.gpu)
                        if ch == 0:
                            h_st = 0
                            h_en = self.args.patch_size 
                        elif ch == ch_range - 1:
                            h_st = h - self.args.patch_size
                            h_en = h
                        else:
                            h_st = ch * (self.args.patch_size - 2 * self.args.edge_size) - self.args.edge_size
                            h_en = h_st + self.args.patch_size
                        for cl in range(cl_range):
                            if cl == 0:
                                l_st = 0
                                l_en = self.args.patch_size
                            elif cl == cl_range - 1:
                                l_st = l - self.args.patch_size
                                l_en = l
                            else:
                                l_st = cl * (self.args.patch_size - 2 * self.args.edge_size) - self.args.edge_size
                                l_en = l_st + self.args.patch_size
                            c_lr = lr[:, :, h_st:h_en, l_st:l_en]

                            c_sr = self.model(c_lr)
                            if cl == 0:
                                c_sr = c_sr[:, :, :, :(self.args.patch_size - 2 * self.args.edge_size)]
                            elif cl == cl_range - 1:
                                c_sr = c_sr[:, :, :, (cl * (self.args.patch_size - 2 * self.args.edge_size) - l):]
                            else:
                                c_sr = c_sr[:, :, :, self.args.edge_size:-self.args.edge_size]

                            tmp_sr = torch.cat((tmp_sr, c_sr), dim=3)
                        if ch == 0:
                            tmp_sr = tmp_sr[:, :, :(self.args.patch_size - 2 * self.args.edge_size), :]
                        elif ch == ch_range - 1:
                            tmp_sr = tmp_sr[:, :, (ch * (self.args.patch_size - 2 * self.args.edge_size) - h):, :]
                        else:
                            tmp_sr = tmp_sr[:, :, self.args.edge_size:-self.args.edge_size, :]

                        sr = torch.cat((sr, tmp_sr), dim=2)
                else:
                    sr = self.model(lr)
                save_results(sr, output_dir, file_name, self.args.scale)

    def is_local_master(self):
        return self.args.local_rank in [-1, 0]
    
    def is_global_master(self):
        return self.args.distributed == False or dist.get_rank() == 0

    def save_checkpoint(self):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint_' + str(self.global_step))
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.model.module.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.bin'))
        if self.enable_GAN:
            torch.save(self.gan_dis_model.state_dict(), os.path.join(output_dir, 'GAN_D_pytorch_model.bin'))
            torch.save(self.gan_dis_optimizer.state_dict(), os.path.join(output_dir, 'GAN_D_optimizer.bin'))

    def main_worker(self):
        if self.is_global_master():
            self.summ_writer.add_text('Train/Args', f"<pre>{json.dumps(self.args.__dict__, indent=2)}</pre>")
        if self.args.gpu is not None:
            logger.info("Use GPU: {} for training.".format(self.args.gpu))

        if self.args.deepspeed:
            self.model, self.optimizer, _, _ = deepspeed.initialize(args=self.args,
                                                                    model=self.model,
                                                                    model_parameters=filter(lambda p: p.requires_grad,
                                                                    self.model.parameters()))
        elif self.args.distributed:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.args.fp16_level)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu], find_unused_parameters=True)
        else:
            if self.args.fp16_level == "O2":
                self.model = self.model.half()
        cudnn.benchmark = True

        if self.args.inference:
            self.inference()
            return

        if self.args.evaluate:
            self.validate()
            return

        total_steps = len(self.train_loader) * self.args.epochs // self.args.gradient_accumulation_steps
        self.optimizer.zero_grad()
        accum_steps = 0
        for epoch in tqdm(range(self.args.epochs), disable=(not self.is_local_master())):
            self.model.train()
            for hr, lr, file_name in tqdm(self.train_loader, disable=(not self.is_local_master())):
                hr = hr.cuda(self.args.gpu, non_blocking=True)
                lr = lr.cuda(self.args.gpu, non_blocking=True)
                sr = self.model(lr)
                loss_vals = {}
                d_loss = None
                for i, l in enumerate(self.losses):
                    if l['function'] is not None:
                        if 'GAN' in l['type']:
                            loss, d_loss = l['function'](sr, hr)
                        else:
                            loss = l['function'](sr, hr)
                        effective_loss = l['weight'] * loss
                        loss_vals[l['type']] = effective_loss
                loss = sum(loss_vals.values())

                accum_steps += 1
                loss = loss / self.args.gradient_accumulation_steps
                if self.args.distributed:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if accum_steps == self.args.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    accum_steps = 0

                    if self.global_step % self.args.log_freq == 0:
                        if self.is_global_master():
                            self.summ_writer.add_scalar('Train/loss_total', loss, self.global_step)
                            self.summ_writer.add_scalar('Train/learning_rate', self.optimizer.get_lr(), self.global_step)
                            self.summ_writer.add_scalar('Train/epoch', epoch, self.global_step)
                            for k, v in loss_vals.items():
                                self.summ_writer.add_scalar('Train/loss_' + k, v, self.global_step)
                            if self.enable_GAN:
                                self.summ_writer.add_scalar('Train/loss_DIS', v, self.global_step)

                    if self.global_step % self.args.eval_freq == 0 and self.is_global_master():
                        self.validate()
                        self.model.train()

                    if self.global_step % self.args.save_freq == 0 and self.is_global_master():
                        self.save_checkpoint()
                    if self.global_step >= total_steps:
                        break
 
            # Tune learning rate by epochs
            self.optimizer.schedule()
            if self.enable_GAN:
                self.gan_dis_optimizer.schedule()
            '''
            if self.is_global_master():
                self.validate()
                self.save_checkpoint()
            '''
            if self.global_step >= total_steps:
                break


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--train_datadir', metavar='DIR', help='path to dataset')
    parser.add_argument('--train_datadir_lr', metavar='DIR', help='path to dataset')
    parser.add_argument('--train_image_ext', type=str, default='.png', help='training image extensions.')
    parser.add_argument('--eval_datadir', metavar='DIR', help='path to dataset')
    parser.add_argument('--eval_datadir_lr', metavar='DIR', help='path to dataset')
    parser.add_argument('--eval_image_ext', type=str, default='.png', help='eval image extensions.')
    parser.add_argument('--noise_datadir', metavar='DIR', help='path to dataset')
    parser.add_argument('--output_dir', default = './', type=str, help='path to save model output.')

    parser.add_argument('--arch', metavar='ARCH', default='drln', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--log_freq', default=8, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save_freq', default=10000, type=int,
                        metavar='N', help='print frequency (default: 10000)')
    parser.add_argument('--eval_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 1000)')

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
    parser.add_argument('--edge_size', type=int, default=0,
                        help='output patch size')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='chances for add noise kernels')
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
    parser.add_argument('--decay', type=str, default='30-60-90',
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
    parser.add_argument('--discriminator_type', type=str, default='NLayerDiscriminator',
                        help='Discriminator type.')
    parser.add_argument('--gan_k', type=int, default=1,
                        help='k value for adversarial loss')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, 
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--loss', type=str, default='1*L1',
                        help='loss function configuration')
    parser.add_argument('--quality_range', type=str, default=None,
                        help='quality range of the LR.')
    parser.add_argument('--blur_types', type=str, default=None,
                        help='quality range of the LR.')

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
