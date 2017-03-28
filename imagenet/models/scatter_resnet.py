import math
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from nested_dict import nested_dict
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ScatResNet(nn.Module):
    def __init__(self,J=3,N=224, num_classes=1000):
        super(ScatResNet, self).__init__()
                
        print(J)
        self.nspace = N / (2 ** J)
        self.nfscat = (1 + 8 * J + 8 * 8 * J * (J - 1) / 2)
        self.ichannels = 256
        self.ichannels2 = 512
        self.inplanes = self.ichannels
        print(self.nfscat)
        print(self.nspace)
        self.bn0 = nn.BatchNorm2d(3*self.nfscat,eps=1e-5, momentum=0.9, affine=False)
        self.conv1 = nn.Conv2d(3*self.nfscat, self.ichannels, kernel_size=3,padding=1)#conv3x3_3D(self.nfscat,self.ichannels)
        self.bn1 = nn.BatchNorm2d(self.ichannels)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.layer1 = self._make_layer(BasicBlock, self.ichannels, 2)
        self.layer2 = self._make_layer(BasicBlock, self.ichannels2, 2, stride=2)
        self.avgpool = nn.AvgPool2d(self.nspace/2)
        
        self.fc = nn.Linear(self.ichannels2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if(m.affine):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes ,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes 
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 3*self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)
      #  x = x.view(x.size(0), 3,self.nfscat, self.nspace, self.nspace)
       # x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.view(x.size(0), self.ichannels, self.nspace, self.nspace)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

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


from torch.nn import Parameter
def params_stats(mod):
    params = OrderedDict()
    stats = OrderedDict()
    for k, v in state_dict(mod).iteritems():
        if isinstance(v, Variable):
            params[k] = v
        else:
            stats[k] = v
    return params,stats

def scatresnet6_2(N,J):

    """Constructs a Scatter + ResNet-10 model.
    Args:
        N: is the crop size (normally 224)
	J: scattering scale (normally 3,4,or 5 for imagenet)
    """
    model = ScatResNet(J,N)
    model=model.cuda()
    params,stats=params_stats(model)

    return model, params, stats

#model,a,b=scatresnet6(4,2)


