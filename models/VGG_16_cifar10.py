import os
import numpy as np

import torch
import torch.nn as nn
#import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


VGG_16_cifar10 = nn.Sequential( # Sequential,
	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2)),
	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2)),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.Dropout(0.1),
	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
	nn.ReLU(),
	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential( # Sequential,
		nn.Dropout(0.1),
		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,512,bias=False)), # Linear,
		nn.ReLU(),
		nn.Dropout(0.1),
		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,10,bias=False)), # Linear,
	),
)


################# Spiking VGG-net: uses spike relus #################################

#from . import spiking_activations
#SpikeRelu = spiking_activations.SpikeRelu
#
##def VGG16_cifar10_block_spike(thresholds, clp_slp):
##class VGG16_cifar10_block_spike(nn.Module):
##    def __init__(self, thresholds, clp_slp=0):
##        super(VGG16_cifar10_block_spike, self).__init__()
##
##        self.VGG16_cifar10_spike = nn.Sequential( # Sequential,
#clp_slp = 0
#out_dir = './vgg_cifar10'
#thresholds = np.loadtxt(os.path.join(out_dir, 'thresholds.txt'))
#VGG16_cifar10_spike = nn.Sequential( # Sequential,
#	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[0], 0, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[1], 1, clp_slp),
#
#	nn.AvgPool2d((2, 2),(2, 2)),
#        SpikeRelu(thresholds[2], 2, clp_slp),
#
#	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[3], 3, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[4], 4, clp_slp),
#
#	nn.AvgPool2d((2, 2),(2, 2)),
#        SpikeRelu(thresholds[5], 5, clp_slp),
#
#	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[6], 6, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[7], 7, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[8], 8, clp_slp),
#
#	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
#        SpikeRelu(thresholds[9], 9, clp_slp),
#
#	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[10], 10, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[11], 11, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[12], 12, clp_slp),
#
#	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
#        SpikeRelu(thresholds[13], 13, clp_slp),
#
#	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[14], 14, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[15], 15, clp_slp),
#	nn.Dropout(0.1),
#
#	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
#        SpikeRelu(thresholds[16], 16, clp_slp),
#
#	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
#        SpikeRelu(thresholds[17], 17, clp_slp),
#
#	Lambda(lambda x: x.view(x.size(0),-1)), # View,
#	nn.Sequential( # Sequential,
#		nn.Dropout(0.1),
#		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,512,bias=False)), # Linear,
#                SpikeRelu(thresholds[18], 18, clp_slp),
#
#		nn.Dropout(0.1),
#		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,10,bias=False)), # Linear,
#	),
#)
#        #return VGG16_cifar10_spike
#    #def forward(self, x):
#    #    out = self.VGG16_cifar10_spike(x)
#    #    return out
#
#'''
#class VGG16_spike(nn.Module):
#
#    def __init__(self, thresholds, clp_slp=0, num_classes=10):
#        super(VGG16_spike, self).__init__()
#        self.layers = VGG16_cifar10_block_spike(thresholds, clp_slp)
#
#    def forward(self, x):
#        x = self.layers(x)
#
#        return x
#'''
