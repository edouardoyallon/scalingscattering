import math
import torch
import torch.nn.functional as F
import torch.cuda.comm as comm
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import cv2
import torchvision
import torchvision.transforms
import torchvision.datasets
import cvtransforms
import torchnet as tnt
import numpy as np
import h5py
import argparse

import os


def build_batch_norm_scattering(path):
    data = torch.load(path)
    EX_ = data['EX']
    EX2_ = data['EX2']
    v = np.sqrt(np.diag(EX2_) - np.power(EX_, 2)+1e-5)
    m = EX_

    m = torch.from_numpy(m).float().cuda()
    m=m.view(1,m.size(0),1,1)

    v = torch.from_numpy(v).float().cuda()
    v = v.view(1, v.size(0), 1, 1)

    m = Variable(m,requires_grad = False)
    v = Variable(v, requires_grad=False)

    return m,v

def batch_norm_scattering(x, m,v):
    m=m.expand_as(x)
    v=v.expand_as(x)
    x = torch.div(torch.add(x,-m),v)
    return x


from collections import OrderedDict
##used to convert nn  modules  for use with SergeyTheano
def state_dict(module, destination=None, prefix=''):
    if destination is None:
        destination = OrderedDict()
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param
    for name, buf in module._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf
    for name, module in module._modules.items():
        if module is not None:
            state_dict(module, destination, prefix + name + '.')
    return destination


from torch.autograd import Variable
def params_stats(mod):
    params = OrderedDict()
    stats = OrderedDict()
    for k, v in state_dict(mod).iteritems():
        if isinstance(v, Variable):
            params[k] = v
        else:
            stats[k] = v
    return params,stats
    

def get_iterator(mode,opt):
    if (opt.imagenetpath is None):
        raise (RuntimeError('Where is imagenet?'))
    if (opt.N is None):
        raise (RuntimeError('Crop size not provided'))
    if (opt.batchSize is None):
        raise (RuntimeError('Batch Size not provided '))
    if (opt.nthread is None):
        raise (RuntimeError('num threads?'))


    def cvload(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32) / 255.0,
        cvtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        lambda x: x.transpose(2, 0, 1).astype(np.float32),
        torch.from_numpy,
    ])




    print("| setting up data loader...")
    if mode:
        traindir = os.path.join(opt.imagenetpath, 'train')
        if (opt.max_samples > 0):
            ds = datasmall.ImageFolder(traindir, tnt.transform.compose([
                cvtransforms.RandomSizedCrop(opt.N),
                cvtransforms.RandomHorizontalFlip(),
                convert,
            ]), loader=cvload,maxSamp=opt.max_samples)
        else:
            ds =torchvision.datasets.ImageFolder(traindir, tnt.transform.compose([
            cvtransforms.RandomSizedCrop(opt.N),
            cvtransforms.RandomHorizontalFlip(),
            convert,
            ]), loader=cvload)
    else:
        if opt.N==224:
            crop_scale=256
        else:
            crop_scale=256*opt.N/224
            
        valdir = os.path.join(opt.imagenetpath, 'val')
        ds = torchvision.datasets.ImageFolder(valdir, tnt.transform.compose([
            cvtransforms.Scale(crop_scale),
            cvtransforms.CenterCrop(opt.N),
            convert,
        ]), loader=cvload)
        


    return torch.utils.data.DataLoader(ds,
                                       batch_size=opt.batchSize, shuffle=mode,
                                       num_workers=opt.nthread, pin_memory=False)



