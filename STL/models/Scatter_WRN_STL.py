import torch.nn as nn
import math
import torch.nn.functional as F
from nested_dict import nested_dict
from collections import OrderedDict
from torch.autograd import Variable
__all__=['resnet16_16_scat_STL','resnet16_8_scat_STL','resnet16_2_scat_STL']
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
     #   self.d = nn.Dropout2d(p=0.2)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

      #  out = self.d(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block,J=2,N=32, k=1,n=2,num_classes=10):
        self.inplanes = 32*k
        self.ichannels = 32*k
        super(ResNet, self).__init__()
        self.nspace = N / (2 ** J)
        self.nfscat = (1 + 8 * J + 8 * 8 * J * (J - 1) / 2)
        self.bn0 = nn.BatchNorm2d(self.nfscat*3,eps=1e-5,affine=False)
        self.conv1 = nn.Conv2d(self.nfscat*3,self.ichannels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.ichannels)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer2 = self._make_layer(block, 32*k, n)
        self.layer3 = self._make_layer(block, 64*k, n,stride=2)
        self.avgpool = nn.AvgPool2d(12)
        self.fc = nn.Linear(64*k, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 3*self.nfscat, self.nspace, self.nspace)
        x = self.bn0(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        

        x = self.layer2(x)
        x = self.layer3(x)
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

def resnet16_64_scat_STL(N,J):
    return resnet16_scat(J,N,2,64)   
def resnet16_16_scat_STL(N,J):
    return resnet16_scat(J,N,2,16)   
def resnet16_8_scat_STL(N,J):
    return resnet16_scat(J,N,2,8)    

def resnet16_2_scat_STL(N,J):
    return resnet16_scat(J,N,2,2) 
    
def resnet16_1_scat_STL(N,J):
    return resnet16_scat(J,N,2,1)
    
def resnet16_scat(J,N,n,k):
    """Constructs a Scat ResNet
      N - crop size
      J - scattering scale
      k - width factor
      n - number of blocks
    """
    model = ResNet(BasicBlock,J,N,k,n)
    
    model=model.cuda()
    params,stats=params_stats(model)

    return model, params, stats
