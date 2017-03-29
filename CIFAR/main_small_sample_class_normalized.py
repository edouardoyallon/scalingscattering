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
import torchvision.datasets as datasets
from scatwave.scattering import Scattering
import torchnet as tnt
import models
import cv2
import argparse
import cvtransforms
  
def parse():
    parser = argparse.ArgumentParser(description='Scattering on CIFAR')
    # Model options
    parser.add_argument('--nthread', default=6, type=int)
  
  
    # Training options
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--sampleSize', default=500, type=int)
    parser.add_argument('--mul', default=20, type=int)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--batchSize', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--weightDecay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                        help='json list with epochs to drop lr on')
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--resume', default='', type=str)
  
    # Save options
    parser.add_argument('--save', default='', type=str,
                        help='save parameters and logs in this folder')
    parser.add_argument('--frequency_save', default=1000,
                        type=int,
                        help='Frequency at which one should save')
    parser.add_argument('--frequency_test', default=10,
                        type=int,
                        help='Frequency at which one should save')
  
    parser.add_argument('--ngpu', default=1, type=int,
                        help='number of GPUs to use for training')
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--scat', default=2, type=int,
                        help='scattering scale, j=0 means no scattering')
    parser.add_argument('--N', default=32, type=int,
                        help='size of the crop')
    parser.add_argument('--model', default='scatfc', type=str,
                        help='name of define of the model in models')
    parser.add_argument('--randomcrop_pad', default=4, type=float)
  
  
    # Display options
    parser.add_argument('--frequency_print', default=10,
                        type=int,
                        help='Frequency at which one should save')
    return parser
  
  
cudnn.benchmark = True
 
parser = parse()
opt = parser.parse_args()
opt.frequency_test=opt.frequency_test*opt.mul
 
  
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
torch.randn(8).cuda()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
epoch_step = list(np.array(json.loads(opt.epoch_step))*opt.mul)
opt.epochs=opt.epochs*opt.mul
 
print('parsed options:', vars(opt))
data_time = 1
  
print(epoch_step)
def create_dataset(opt, mode):
    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),
        lambda x: x / 255.0,
        # cvtransforms.Normalize([125.3, 123.0, 113.9], [63.0,  62.1,  66.7]),
        lambda x: x.transpose(2, 0, 1).astype(np.float32),
        torch.from_numpy,
    ])
  
    train_transform = tnt.transform.compose([
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.Pad(opt.randomcrop_pad, cv2.BORDER_REFLECT),
        cvtransforms.RandomCrop(32),
        convert,
    ])
 
     
    ds = getattr(datasets, opt.dataset)('.', train=mode, download=True)
    smode = 'train' if mode else 'test'
    if mode:
        from numpy.random import RandomState
        prng = RandomState(opt.seed)
         
        assert(opt.sampleSize%10==0)
         
        random_permute=prng.permutation(np.arange(0,5000))[0:opt.sampleSize/10]

        labels = np.array(getattr(ds,'train_labels'))
        data = getattr(ds,'train_data')
       
        classes=np.unique(labels)
        inds_all=np.array([],dtype='int32')
        for cl in classes:
            inds=np.where(np.array(labels)==cl)[0][random_permute]
            inds_all=np.r_[inds,inds_all]

        ds = tnt.dataset.TensorDataset([
            data[inds_all,:].transpose(0, 2, 3, 1),
            labels[inds_all].tolist()])
    else:
        ds = tnt.dataset.TensorDataset([
            getattr(ds, smode + '_data').transpose(0, 2, 3, 1),
            getattr(ds, smode + '_labels')])
    return ds.transform({0: train_transform if mode else convert})
  
  
def main():
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)
  
    if opt.scat>0:
        model, params, stats = models.__dict__[opt.model](N=opt.N,J=opt.scat)
    else:
        model, params, stats = models.__dict__[opt.model]()
  
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = %f'% lr)
        return torch.optim.SGD(params.values(), lr, opt.momentum, weight_decay=opt.weightDecay)
    def get_iterator(mode):
        ds = create_dataset(opt, mode)
        return ds.parallel(batch_size=opt.batchSize, shuffle=mode,
                           num_workers=opt.nthread, pin_memory=False)
  
    optimizer = create_optimizer(opt, opt.lr)
  
    iter_test = get_iterator(False)
    iter_train = get_iterator(True)
  
    if opt.scat>0:
        scat = Scattering(M=opt.N, N=opt.N, J=opt.scat, pre_pad=False).cuda()
  
    epoch = 0
    if opt.resume != '':
        resumeFile=opt.resume
        if not resumeFile.endswith('pt7'):
            resumeFile=torch.load(opt.resume + '/latest.pt7')['latest_file']
            state_dict = torch.load(resumeFile)
            epoch = state_dict['epoch']
            params_tensors, stats = state_dict['params'], state_dict['stats']
            for k, v in params.iteritems():
                v.data.copy_(params_tensors[k])
            optimizer.load_state_dict(state_dict['optimizer'])
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
        y = torch.nn.parallel.data_parallel(model, inputs, np.arange(opt.ngpu).tolist())
        return F.cross_entropy(y, targets), y
  
  
    def log(t, state):
        if(t['epoch']>0 and t['epoch']%opt.frequency_save==0):
            torch.save(dict(params={k: v.data.cpu() for k, v in params.iteritems()},
                        stats=stats,
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   open(os.path.join(opt.save, 'epoch_%i_model.pt7' % t['epoch']), 'w'))
            torch.save( dict(latest_file=os.path.join(opt.save, 'epoch_%i_model.pt7' % t['epoch'])
                            ),
                        open(os.path.join(opt.save, 'latest.pt7'), 'w'))
  
        z = vars(opt).copy()
        z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)
  
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
        if(state['t']%opt.frequency_print==0 and state['t']>0):
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
  

        state['iterator'] = iter_train
  
        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            print('changing LR')
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)
  
    def on_end_epoch(state):
        if(state['t']%opt.frequency_test==0 and state['t']>0):
            train_loss = meter_loss.value()
            train_acc = classacc.value()
            train_time = timer_train.value()
            meter_loss.reset()
            classacc.reset()
            timer_test.reset()
      
            engine.test(h, iter_test)
      
            log({
                "train_loss": train_loss[0],
                "train_acc": 100-train_acc[0],
                "test_loss": meter_loss.value()[0],
                "test_acc": 100-classacc.value()[0],
                "epoch": state['epoch'],
                "n_parameters": n_parameters,
                "train_time": train_time,
                "test_time": timer_test.value(),
            }, state)
  
  
  
  
    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, iter_train, opt.epochs, optimizer)
  
  
if __name__ == '__main__':
    main()
