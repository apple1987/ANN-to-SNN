import os
import errno
import sys
import time
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def float_to_fixed(val, W=2, F=6):
    ''' converts floating point number val to fixed format W.F
    using method described in http://ee.sharif.edu/~asic/Tutorials/Fixed-Point.pdf '''
    nearest_int = torch.round(val*(2**F))
    return nearest_int*(1.0/2**F)


def quantize_model(model, W=2, F=6):

    ''' Quantize the weights of a model

    Args:
        model: a pytorch model class
        W: num bits for integer part
        F: num bits for fractional part
    '''

    if has_bn(model):
        print('\n\n[Warning]: Model has BN layers!!\n\n')
    quantized_model = model

    # Serialize the original model
    name_to_type = serialize_model(quantized_model)

    for n, t in name_to_type.items():
        if isinstance(t, nn.Conv2d):
            wts = t.weight.data
            print('pre wt: min: {}\tmax: {}'.format(torch.min(wts), torch.max(wts)))
            t.weight.data = float_to_fixed(wts, W, F)
            print('post wt: min: {}\tmax: {}'.format(torch.min(t.weight.data), torch.max(t.weight.data)))

            if t.bias is not None:
                bias = t.bias.data
                print('pre bias: min: {}\tmax: {}'.format(torch.min(bias), torch.max(bias)))
                t.bias.data = float_to_fixed(bias, W, F)
                print('post bias: min: {}\tmax: {}'.format(torch.min(t.bias.data), torch.max(t.bias.data)))

    return quantized_model



def serialize_model(model):
    "gives relative ordering of layers in a model:"
    "layer-name => layer-type"

    name_to_type = OrderedDict()
    layer_num = 0
    for name, module in model.named_modules():
        #print(name)
        if isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ReLU) or \
                isinstance(module, nn.Linear) or \
                isinstance(module, nn.AvgPool2d) or \
                isinstance(module, nn.BatchNorm2d) or \
                isinstance(module, nn.Dropout):

            name_to_type[name] = module
            layer_num += 1

    return name_to_type


def adjust_weights(wt_layer, bn_layer):
    num_out_channels = wt_layer.weight.size()[0]

    bias = torch.zeros(num_out_channels)
    wt_layer_bias = torch.zeros(num_out_channels)
    if wt_layer.bias is not None:
        wt_layer_bias = wt_layer.bias

    wt_cap = torch.zeros(wt_layer.weight.size())
    for i in range(num_out_channels):
        gamma = bn_layer.weight[i]
        beta = bn_layer.bias[i]
        sigma = bn_layer.running_var[i]
        mu = bn_layer.running_mean[i]
        eps = bn_layer.eps
        scale_fac = gamma / torch.sqrt(eps+sigma)
        wt_cap[i,:,:,:] = wt_layer.weight[i,:,:,:]*scale_fac
        bias[i] = (wt_layer_bias[i]-mu)*scale_fac + beta
    return (wt_cap, bias)


def merge_bn(model, model_nobn):

    "merges bn params with those of the previous layer"
    "works for the layer pattern: conv->bn only"

    # Serialize the original model
    name_to_type = serialize_model(model)
    key_list = list(name_to_type.keys())

    # Serialize the nobn model
    name_to_type_nobn = serialize_model(model_nobn)


    for i,n in enumerate(key_list):
        if isinstance(name_to_type[n], nn.Conv2d) and \
                isinstance(name_to_type[key_list[i+1]], nn.BatchNorm2d):

            conv_layer = name_to_type[n]
            bn_layer = name_to_type[key_list[i+1]]
            new_wts, new_bias = adjust_weights(conv_layer, bn_layer)

            conv_layer_nobn = name_to_type_nobn[n]
            conv_layer_nobn.weight.data = new_wts
            if conv_layer_nobn.bias is not None:
                conv_layer_nobn.bias.data = new_bias

        elif isinstance(name_to_type[n], nn.Conv2d) or \
                isinstance(name_to_type[n], nn.Linear):
            layer = name_to_type[n]
            layer_nobn = name_to_type_nobn[n]
            layer_nobn.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                layer_nobn.bias.data = layer.bias.data.clone()

    return model_nobn


def has_bn(net):
    for m in net.modules():
        if type(m) == nn.BatchNorm2d:
            return True
    return False


def validate(net, testloader, device='cuda:0'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), acc, correct, total))
                #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return acc


def load_cifar10(data_dir='./data', arch='mobilenet_cifar10'):
    # Data
    print('==> Preparing data..')
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if arch == 'vgg_cifar10':
        std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainloader, testloader)


def save_model(net, state, model_path, file_name):
    assert os.path.isdir(model_path), 'Error: no {} directory found!'.format(model_path)
    file_path = os.path.join(model_path, file_name)
    print('Saving..')
    state['net'] = net.state_dict()
    torch.save(state, file_path)


def load_model(net, model_path, file_name):
    # Load checkpoint.
    file_path = os.path.join(model_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(file_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print (best_acc, start_epoch)
    return checkpoint, net


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
term_width = '143' # note: edit by sms821 for nohup to work
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
