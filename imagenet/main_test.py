import os
import re
import json
import numpy as np
 
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import torch
 
from torchnet import dataset, meter
from torchnet.engine import Engine
 
from torch.autograd import Variable
from torch.nn import Parameter
from torch.backends import cudnn
import torch.nn.functional as F
from utils import get_iterator
 
#from scattering import Scattering
from scatwave.scattering import Scattering
 
import models
import argparse
 
 
def parse():
    parser = argparse.ArgumentParser(description='Scattering on Imagenet')
    # Model options
    parser.add_argument('--imagenetpath', default='/media/ssd/dataset/', type=str)
    parser.add_argument('--nthread', default=4, type=int)
    parser.add_argument('--resume', default='scatter_resnet_10_model.pt7', type=str)
 
    parser.add_argument('--ngpu', default=1, type=int,
                        help='number of GPUs to use for training')
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--scat', default=3, type=int,
                        help='scattering scale, j=0 means no scattering')
    parser.add_argument('--N', default=224, type=int,
                        help='size of the crop')
    parser.add_argument('--model', default='scatresnet6_2', type=str,
                        help='name of define of the model in models')
    parser.add_argument('--batchSize', default=256, type=int)
    return parser
 
 
 
 
 
cudnn.benchmark = True
 
parser = parse()
opt = parser.parse_args()
print('parsed options:', vars(opt))
 
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
torch.randn(8).cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
 
data_time = 1
 
 
def main():
    model, params, stats = models.__dict__[opt.model](N=opt.N,J=opt.scat)
 
    iter_test = get_iterator(False, opt)
 
    scat = Scattering(M=opt.N, N=opt.N, J=opt.scat, pre_pad=False).cuda()
 
    epoch = 0
    if opt.resume != '':
        resumeFile=opt.resume
        if not resumeFile.endswith('pt7'):
            resumeFile=torch.load(opt.resume + '/latest.pt7')['latest_file']
        state_dict = torch.load(resumeFile)
        
        model.load_state_dict(state_dict['state_dict']) 
        print('model was restored from epoch:',epoch)
 
    print('\nParameters:')
    print(pd.DataFrame([(key, v.size(), torch.typename(v.data)) for key, v in params.items()]))
    print('\nAdditional buffers:')
    print(pd.DataFrame([(key, v.size(), torch.typename(v)) for key, v in stats.items()]))
    n_parameters = sum([p.numel() for p in list(params.values()) + list(stats.values())])
    print('\nTotal number of parameters: %f'% n_parameters)
 
    meter_loss = meter.AverageValueMeter()
    classacc = meter.ClassErrorMeter(topk=[1, 5], accuracy=False)
    timer_data = meter.TimeMeter('s')
    timer_sample = meter.TimeMeter('s')
    timer_train = meter.TimeMeter('s')
    timer_test = meter.TimeMeter('s')
 
 
    def h(sample):
        inputs = sample[0].cuda()
        if opt.scat > 0:
            inputs = scat(inputs)
        inputs = Variable(inputs)
        targets = Variable(sample[1].cuda().long())
        if sample[2]:
            model.train()
        else:
            model.eval()
       # y = model.forward(inputs)
        y = torch.nn.parallel.data_parallel(model, inputs, np.arange(opt.ngpu).tolist())
        return F.cross_entropy(y, targets), y
 
    def on_sample(state):
        global data_time
        data_time = timer_data.value()
        timer_sample.reset()
        state['sample'].append(state['train'])
 
 
    def on_forward(state):
        prev_sum5=classacc.sum[5]
        prev_sum1 = classacc.sum[1]
        classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])
 
        next_sum5 = classacc.sum[5]
        next_sum1 = classacc.sum[1]
        n =  state['output'].data.size(0)
        curr_top5=100.0*(next_sum5-prev_sum5)/n
        curr_top1 = 100.0*(next_sum1 - prev_sum1) / n
        sample_time = timer_sample.value()
        timer_data.reset()
        if(state['train']):
            txt = 'Train:'
        else:
            txt = 'Test'
        
        print('%s [%i,%i/%i] ; loss: %.3f (%.3f) ; acc5: %.2f (%.2f) ; acc1: %.2f (%.2f) ; data %.3f ; time %.3f' %
                  (txt, state['epoch'],state['t']%len(state['iterator']),
                   len(state['iterator']),
                   state['loss'].data[0],
                   meter_loss.value()[0],
                   curr_top5,
                   classacc.value(5),
                   curr_top1,
                   classacc.value(1),
                   data_time,
                   sample_time
 
                   ))
 
 
    def on_start(state):
        state['epoch'] = epoch
 
    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
 
        epoch = state['epoch'] + 1
        
    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()
 
        engine.test(h, iter_test)
 
 
 
 
 
    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.test(h, iter_test)
    print(classacc.value())
 
if __name__ == '__main__':
    main()
