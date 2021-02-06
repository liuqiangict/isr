import os
import sys
import json
import collections
import unicodedata
import torch
import math
import re
import logging
import numpy as np
import imageio
import base64
import io
import uuid

import PIL
from PIL import Image

import models
import time
import datetime
import utils
import json

logger = logging.getLogger(__name__)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class ISRService:
    def __init__(self):
        super().__init__()

        self.edge_size = 0
        self.patch_size = 144
        base_dir = utils.get_model_path()
        self.optim_model_path = os.path.join(os.path.dirname(os.path.realpath(base_dir)), "model/model_optim")
        self.scaleup_model_path = os.path.join(os.path.dirname(os.path.realpath(base_dir)), "model/model_scaleup")
        
        self.optim_model = self.build_model('longformer', self.optim_model_path)
        self.optim_model.eval()
        self.scaleup_model = self.build_model('drln', self.scaleup_model_path, False)
        self.scaleup_model.eval()

    def build_model(self, arch, model_path, is_half=True):
        logger.info("Creating model '{}'".format(arch))
        model = models.__dict__[arch](model_path)

        model_path = os.path.join(model_path, 'pytorch_model.bin')
        checkpoint = torch.load(model_path, 'cpu')
        model.load_state_dict(checkpoint, strict=True)

        if is_half:
            return model.cuda().half()
        else:
            return model.cuda()

    def inference_optim(self, lr):
        with torch.no_grad():
            sr = torch.tensor([], dtype=lr.dtype).cuda()
            b, c, h, l = lr.shape
            ch_range = h // (self.patch_size - 2 * self.edge_size)
            if h % (self.patch_size - 2 * self.edge_size) != 0:
                ch_range += 1
            cl_range = l // (self.patch_size - 2 * self.edge_size)
            if l % (self.patch_size - 2 * self.edge_size) != 0:
                cl_range += 1
            for ch in range(ch_range):
                tmp_sr = torch.tensor([], dtype=lr.dtype).cuda()
                if ch == 0:
                    h_st = 0
                    h_en = self.patch_size 
                elif ch == ch_range - 1:
                    h_st = h - self.patch_size
                    h_en = h
                else:
                    h_st = ch * (self.patch_size - 2 * self.edge_size) - self.edge_size
                    h_en = h_st + self.patch_size
                for cl in range(cl_range):
                    if cl == 0:
                        l_st = 0
                        l_en = self.patch_size
                    elif cl == cl_range - 1:
                        l_st = l - self.patch_size
                        l_en = l
                    else:
                        l_st = cl * (self.patch_size - 2 * self.edge_size) - self.edge_size
                        l_en = l_st + self.patch_size
                    c_lr = lr[:, :, h_st:h_en, l_st:l_en]

                    c_sr = self.optim_model(c_lr)
                    if cl == 0:
                        c_sr = c_sr[:, :, :, :(self.patch_size - 2 * self.edge_size)]
                    elif cl == cl_range - 1:
                        c_sr = c_sr[:, :, :, (cl * (self.patch_size - 2 * self.edge_size) - l):]
                    else:
                        c_sr = c_sr[:, :, :, self.edge_size:-self.edge_size]

                    tmp_sr = torch.cat((tmp_sr, c_sr), dim=3)
                if ch == 0:
                    tmp_sr = tmp_sr[:, :, :(self.patch_size - 2 * self.edge_size), :]
                elif ch == ch_range - 1:
                    tmp_sr = tmp_sr[:, :, (ch * (self.patch_size - 2 * self.edge_size) - h):, :]
                else:
                    tmp_sr = tmp_sr[:, :, self.edge_size:-self.edge_size, :]

                sr = torch.cat((sr, tmp_sr), dim=2)

        return sr

    def inference_scaleup(self, lr):
        with torch.no_grad():
            sr = self.scaleup_model(lr)
        return sr


    def inference(self, line):
        data = json.loads(line)

        size = len(data['names'])
        names = data['names']
        imgs = []
        for b64_img in data['b64_imgs']:
            img = imageio.imread(io.BytesIO(base64.b64decode(b64_img)))
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis=-1)
            img = img[:, :, :3]
            imgs.append(img)

        lr = torch.tensor(imgs, dtype=torch.float).cuda().permute(0, 3, 1, 2).half()

        optim_sr = self.inference_optim(lr)
        # post process for optim model
        # pre-process for scale up model
        scaleup_sr = self.inference_scaleup(optim_sr.round().to(torch.float32))

        optim_img = optim_sr.clamp_(0, 255).round().byte().permute(0, 2, 3, 1).cpu().detach().numpy()
        scaleup_img = scaleup_sr.clamp_(0, 255).round().byte().permute(0, 2, 3, 1).cpu().detach().numpy()
        res = {'names': [], 'intermediate_res': [], 'res': []}
        for i in range(size):
            optim_buf = io.BytesIO()
            imageio.imwrite(optim_buf, optim_img[i], format='jpg')
            scaleup_buf = io.BytesIO()
            imageio.imwrite(scaleup_buf, scaleup_img[i], format='jpg')

            res['names'].append(names[i])
            res['intermediate_res'].append(base64.b64encode(optim_buf.getvalue()).decode())
            res['res'].append(base64.b64encode(scaleup_buf.getvalue()).decode())

        return json.dumps(res)
    
    def Inference(self, line):
        return self.inference(line)

    def eval(self, line):
        return self.inference(line)

    def Eval(self, line):
        return self.inference(line)


def main():

    model_instance = ISRService()
    #filename = "/relevance-nfs/users/qiangliu/TISR/Imagery_75/eval/eval_18_19/demo/LR/021211312300311113.jpg"
    #with open(filename, "rb") as fid:
    #    img = fid.read()
    #b64_img = base64.b64encode(img).decode()
    #b64_res = model_instance.inference(b64_img)
    #img = imageio.imread(io.BytesIO(base64.b64decode(b64_res)))
    #imageio.imwrite('res.png', img)
    #with open('out.txt', "w", encoding='utf-8') as writer:
    #    writer.write(b64_res + '\n')

    inputs = []
    with open('input_144_batch_2_json.txt', mode='r', encoding='utf-8') as reader:
        for i, line in enumerate(reader):
            if i > 4:
                break
            inputs.append(line.strip())

    outputs = []
    for i, line in enumerate(inputs): 
        print(datetime.datetime.now(), i)
        res = model_instance.inference(line)
        outputs.append(res)

    with open('output.txt', mode='w', encoding='utf-8') as writer:
        for b64_res in outputs:
            writer.write(b64_res + '\n')

if __name__ == '__main__':
    main()

