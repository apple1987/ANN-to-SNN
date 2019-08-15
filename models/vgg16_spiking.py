import torch
import torch.nn as nn

from . import spiking_activations
SpikeRelu = spiking_activations.SpikeRelu

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


def vgg16_spike(thresholds, clp_slp=0, num_classes=10, device='cuda:0', reset='to-threshold'):
    #sr0 = SpikeRelu(thresholds[0], 0, clp_slp, device, reset), #1
    #print(type(sr0))

    VGG16_cifar10_spike = nn.Sequential( # Sequential,
    	nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            #sr0(thresholds[0], 0, clp_slp, device, reset), #1
            SpikeRelu(thresholds[0], 0, clp_slp, device, reset), #1
    	nn.Dropout(0.1),

    	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[1], 1, clp_slp, device, reset), #2

    	nn.AvgPool2d((2, 2),(2, 2)),
            SpikeRelu(thresholds[2], 2, clp_slp, device, reset), #3

    	nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[3], 3, clp_slp, device, reset), #4
    	nn.Dropout(0.1),

    	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[4], 4, clp_slp, device, reset), #5

    	nn.AvgPool2d((2, 2),(2, 2)),
            SpikeRelu(thresholds[5], 5, clp_slp, device, reset), #6

    	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[6], 6, clp_slp, device, reset), #7
    	nn.Dropout(0.1),

    	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[7], 7, clp_slp, device, reset), #8
    	nn.Dropout(0.1),

    	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[8], 8, clp_slp, device, reset), #9

    	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
            SpikeRelu(thresholds[9], 9, clp_slp, device, reset), #10

    	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[10], 10, clp_slp, device, reset), #11
    	nn.Dropout(0.1),

    	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[11], 11, clp_slp, device, reset), #12
    	nn.Dropout(0.1),

    	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[12], 12, clp_slp, device, reset), #13

    	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
            SpikeRelu(thresholds[13], 13, clp_slp, device, reset), #14

    	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[14], 14, clp_slp, device, reset), #15
    	nn.Dropout(0.1),

    	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[15], 15, clp_slp, device, reset), #16
    	nn.Dropout(0.1),

    	nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,1,bias=False),
            SpikeRelu(thresholds[16], 16, clp_slp, device, reset), #17

    	nn.AvgPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),#AvgPool2d,
            SpikeRelu(thresholds[17], 17, clp_slp, device, reset), #18

    	Lambda(lambda x: x.view(x.size(0),-1)), # View,
    	nn.Sequential( # Sequential,
    		nn.Dropout(0.1),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,512,bias=False)), # Linear,
                    SpikeRelu(thresholds[18], 18, clp_slp, device, reset), #19

    		nn.Dropout(0.1),
    		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(512,10,bias=False)), # Linear, #20
                    SpikeRelu(thresholds[19], 19, clp_slp, device, reset), #20
    	),
    )
    return VGG16_cifar10_spike
