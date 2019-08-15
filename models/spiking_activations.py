import torch
import torch.nn as nn

### clamping supported
def SpikeRelu(v_th, layer_num=None, clp_slp=None, dev='cuda:0', reset='to-threshold'):
    #class spikeRelu(nn.Module):

    #    def __init__(self, v_th, layer_num=0, clp_slp=0, dev='cuda:0', reset='to-threshold'):
    #        super(spikeRelu, self).__init__()
    #        #print("initiating spike relu..")
    #        self.threshold = v_th
    #        self.layer_num = layer_num
    #        self.d = clp_slp
    #        self.vmem = 0
    #        self.clamp_time = layer_num * clp_slp

    #    def forward(self, x):
    #        self.clamp_time -= 1

    #        if self.clamp_time <= 0:
    #            # integrate the sum(w_i*x_i)
    #            self.vmem += x

    #        # generate output spikes
    #        op_spikes = torch.where(self.vmem >= self.threshold, torch.ones([1], device=torch.device(dev)), \
    #                torch.zeros([1], device=torch.device(dev)))

    #        # vmem reset
    #        # 'reset-to-zero'
    #        if reset == 'to-zero':
    #            self.vmem = torch.where(self.vmem >= self.threshold, torch.zeros(1, device=torch.device(dev)), self.vmem)

    #        # 'reset-to-threshold'
    #        elif reset == 'to-threshold':
    #            self.vmem = torch.where(self.vmem >= self.threshold, self.vmem-self.threshold, self.vmem)

    #        else:
    #            print('Invalid reset mechanism {}'.format(reset))

    #        return op_spikes

    #    def extra_repr(self):
    #        return 'v_th : {}, vmem : {}'.format(v_th, self.vmem)

    #return spikeRelu
    #print(type(layer_num), type(clp_slp))
    return spikeRelu(v_th, layer_num, clp_slp, dev, reset)


class spikeRelu(nn.Module):

    def __init__(self, v_th, layer_num=0, clp_slp=0, dev='cuda:0', reset='to-threshold'):
        super(spikeRelu, self).__init__()
        #print("initiating spike relu..")
        self.threshold = v_th
        self.layer_num = layer_num
        self.d = clp_slp
        self.vmem = 0
        self.clamp_time = layer_num * clp_slp
        self.dev = dev
        self.reset = reset

    def forward(self, x):
        self.clamp_time -= 1

        if self.clamp_time <= 0:
            # integrate the sum(w_i*x_i)
            self.vmem += x

        # generate output spikes
        op_spikes = torch.where(self.vmem >= self.threshold, torch.ones([1], device=torch.device(self.dev)), \
                torch.zeros([1], device=torch.device(self.dev)))

        # vmem reset
        # 'reset-to-zero'
        if self.reset == 'to-zero':
            self.vmem = torch.where(self.vmem >= self.threshold, torch.zeros(1, device=torch.device(self.dev)), self.vmem)

        # 'reset-to-threshold'
        elif self.reset == 'to-threshold':
            self.vmem = torch.where(self.vmem >= self.threshold, self.vmem-self.threshold, self.vmem)

        else:
            print('Invalid reset mechanism {}'.format(reset))

        return op_spikes

    def extra_repr(self):
        return 'v_th : {}, vmem : {}'.format(self.threshold, self.vmem)
        #return 'v_th : {}, vmem : {}'.format(v_th, self.vmem)
