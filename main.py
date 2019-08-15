import numpy as np
import os
import configparser
import argparse
import pprint

import torch
from torchsummary import summary

from models import MobileNet, MobileNet_nobn
from models import VGG_16_cifar10
from utils import *
from spiking import *

def main():
    parser = argparse.ArgumentParser(description='Deep Learning SNN simulation')
    parser.add_argument('--config-file', default='config_1.ini')
    args = parser.parse_args()
    print (args)

    config = configparser.ConfigParser()
    config.read(args.config_file)
    pprint.pprint({section: dict(config[section]) for section in config.sections()})
    print

    defaults = config['DEFAULT']
    device = defaults['device']
    app_name = defaults['app_name']

    print ('[INFO] Simulating spiking {}'.format(app_name))
    if not os.path.isdir(app_name):
        os.mkdir(app_name)
    out_dir = app_name

    org_model = config['original model']

    " Load the original model "
    net = None
    if 'mobilenet_cifar10' in org_model['arch']:
        net = MobileNet()
    if org_model['arch'] == 'vgg_cifar10':
        net = VGG_16_cifar10

    num_classes = org_model.getint('num_classes')
    model_path = org_model['model_path']
    file_name = org_model['file_name']
    state = None
    state, net = load_model(net, model_path, file_name)
    net = net.to(device)

    " Load the dataset "
    testloader, img_size = None, None
    data_config = config['dataset']
    data_dir = data_config['data_dir']
    if data_config['dataset'] == 'cifar10':
        _, testloader = load_cifar10(data_dir, org_model['arch'])
        img_size = [-1,3,32,32]

    " Tasks to do "
    tasks = config['functions']
    if tasks.getboolean('validate'):
        print(net)
        validate(net, testloader, device)

    new_net = None
    if tasks.getboolean('remove_bn'):
        if has_bn(net):
            new_net = MobileNet_nobn()
            new_net = merge_bn(net, new_net)
            new_net = new_net.to(device)
            print(new_net)
            print('Validating model after folding back BN layers...')
            validate(new_net, testloader, device)
            save_model(new_net, state, out_dir, 'nobn_'+file_name)
        else:
            print('model has no BN layers')

    if tasks.getboolean('use_nobn'):
        if 'mobilenet_cifar10' in org_model['arch']:
            net = MobileNet_nobn()
            state, net = load_model(net, out_dir, 'nobn_'+file_name)
            net = net.to(device)

    " quantize model "
    quant_config = config['quantization']
    W = quant_config.getint('W')
    F = quant_config.getint('F')
    if tasks.getboolean('quantize'):
        net = quantize_model(net, W, F)
        net = net.to(device)
        print('Validating model after quantization...')
        state['acc'] = validate(net, testloader, device)
        save_model(net, state, out_dir, 'quant{}.{}_'.format(W, F)+file_name)

    " use quantized model "
    if tasks.getboolean('use_quant'):
        state, net = load_model(net, out_dir, 'quant{}.{}_'.format(W, F)+file_name)
        net = net.to(device)

    if tasks.getboolean('validate_nobn'):
        if not has_bn(net):
            print('Validating no_bn model...')
            validate(net, testloader)
        else:
            print('model has BN layers!! Exiting..')
            exit()

    " compute thresholds "
    spike_config = config['spiking']
    percentile = spike_config.getfloat('percentile')
    if spike_config.getboolean('compute_thresholds'):
        compute_thresholds(net, testloader, out_dir, percentile, device)

    " convert ann to snn "
    spike_net, thresholds, max_acts, clamp_slope = None, None, None, None
    if spike_config.getboolean('convert_to_spike'):
        thresholds = np.loadtxt(os.path.join(out_dir, 'thresholds.txt'))
        max_acts = np.loadtxt(os.path.join(out_dir, 'max_acts.txt'))
        clamp_slope = spike_config.getfloat('clamp_slope')

        spike_net = createSpikingModel(net, org_model['arch'], num_classes, spike_config, thresholds, max_acts, device)
        print(spike_net)
        sanity_check(net, spike_net, max_acts)

    " simulate snn "
    if spike_config.getboolean('simulate_spiking'):
        thresholds = np.loadtxt(os.path.join(out_dir, 'thresholds.txt'))
        max_acts = np.loadtxt(os.path.join(out_dir, 'max_acts.txt'))

        simulate_spike_model(net, org_model['arch'], testloader, config, thresholds, max_acts, num_classes, img_size, device)

    " plot correlations "
    if spike_config.getboolean('plot_correlations'):
        corr = np.load(os.path.join(out_dir, 'layerwise_corr.npy'))
        plot_config = config['plotting']
        #print('corr matrix shape: {}'.format(corr.shape))
        plot_correlations(corr, out_dir, plot_config)


if __name__ == '__main__':
    main()


def saveData(outFName, spikes, t):
    #outFName = "savedData/mnist_spiking_tw40.h5"
    gpName = "mnist"
    hf = h5py.File(outFName, 'a')
    g = hf.get(gpName)
    if not g:
        g = hf.create_group(gpName)
    dsName = "ts-" + str(t)
    g.create_dataset(dsName, data=spikes)


def create_buffers(probe_layers):
    mats = []
    with open('map.json') as json_file:
        data = json.load(json_file)
        for p in probe_layers:
            #layer_num = int(p[-1])
            list_num = re.findall("[0-9]", p)
            str_num = ''
            for l in list_num:
                str_num += l
            layer_num = int(str_num)

            shape = data[p]
            temp = np.zeros(shape)
            new_shape = [-1]
            for s in shape:
                new_shape.append(s)
            temp = np.reshape(temp, new_shape)
            #temp = np.expand_dims(temp, axis=0)
            mats.append([temp, layer_num])
    return mats

