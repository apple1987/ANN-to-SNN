
import numpy as np

import torch
import torch.nn as nn

from utils import *

import json
import matplotlib
import matplotlib.pyplot as plt

################# Spiking VGG-net: uses spike relus #################################

def simulate_spike_model(net, arch, val_loader, config, thresholds, max_acts, num_classes, img_size, device='cuda:0'):

    out_dir = config['DEFAULT']['app_name']
    spike_config = config['spiking']
    batch_size = spike_config.getint('batch_size')
    clamp_slope = spike_config.getfloat('clamp_slope')
    time_window = spike_config.getint('time_window')
    numBatches = spike_config.getint('num_batches')

    net.eval()

    spike_net = createSpikingModel(net, arch, num_classes, spike_config, thresholds, max_acts, device)
    spike_net.eval()
    if spike_config.getboolean('plot_mean_var'):
        plot_mean_var(net, spike_net, out_dir)

    buffers = None
    hooks = None
    num_layers = None
    layers = []
    if spike_config.getboolean('save_activations'):
        hooks, buffers = create_buffers(net, img_size, device)
        num_layers = len(buffers)
        image_corr = np.zeros(numBatches*batch_size)
        layer_corr = np.zeros((num_layers, numBatches*batch_size))


    total_correct = 0
    expected_correct = 0
    combined_model_correct = 0
    total_images = 0
    batch_num = 0
    confusion_matrix = np.zeros([num_classes,num_classes], int)
    out_spikes_t_b_c = torch.zeros((time_window, batch_size, num_classes))

    spike_buffers = None

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if batch_num >= numBatches:
                break
            print ('\n\n------------ inferring batch {} -------------'.format(batch_num))
            images, labels = data

            # perform inference on the original model to compare
            images = images.to(device)
            net = net.to(device)
            output_org = net(images.float())
            _, predicted_org = torch.max(output_org.data, 1)

            # create the spiking model
            spike_net = createSpikingModel(net, arch, num_classes, spike_config, thresholds, max_acts, device)
            sanity_check(net, spike_net, max_acts)
            #print(spike_net)
            spike_net.eval()

            spike_hooks = None
            if spike_config.getboolean('save_activations'):

                #### record outputs of intermediate layers of original model
                buffers[0] = images
                for z,h in enumerate(hooks):
                    layer_out = h.output
                    buffers[z+1] = layer_out

                #print(device)
                spike_hooks, spike_buffers = create_spike_buffers(spike_net, img_size, device)
                assert(len(buffers) == len(spike_buffers))
                for i in range(len(buffers)):
                    assert(buffers[i].size() == spike_buffers[i].size())

                #print('spike_hooks: {}\thooks: {}'.format(len(spike_hooks), len(hooks)))
                assert(len(spike_hooks) == len(hooks) == num_layers-1)


            # starting of time-stepped spike integration
            for t in range(time_window):

                # convert image pixels to spiking inputs
                spikes = poisson_spikes(images.cpu().numpy())

                # supply random inputs
                spikes = torch.from_numpy(spikes)

                out_spikes = None
                if spike_config.get('poisson_spikes'):

                    # supplying spiking inputs to the spiking model
                    spikes = spikes.to(device)
                    out_spikes = spike_net(spikes.float())
                else:

                    # supplying original analog inputs to the spiking model
                    images = images.to(device)
                    out_spikes = spike_net(images.float())

                out_spikes_t_b_c[t,:,:] = out_spikes

                # record sum of spikes from intermediate layers
                if spike_config.getboolean('save_activations'):
                    for z in range(len(spike_hooks)+1):
                        h = spike_hooks[z-1]
                        if z == 0:
                            spike_buffers[0] += spikes[0].float()
                        else:
                            layer_out = h.output[0]
                            spike_buffers[z] += layer_out

            # end of time-stepped spike integration

            # corresponding analog value of these spikes
            if spike_config.getboolean('save_activations'):
                for z in range(len(spike_hooks)+1):
                    spike_buffers[z] = (spike_buffers[z] / time_window) * max_acts[z]

                # save the correlation coefficients between spike acts and analog acts
                for l in range(len(hooks)+1):
                    if len(buffers[l].size()) > 2:
                        B,C,H,W = buffers[l].size()
                        buffers[l] = buffers[l].view(-1, C*H*W)
                        spike_buffers[l] = spike_buffers[l].view(-1, C*H*W)
                    else:
                        B,C = buffers[l].size()
                        buffers[l] = buffers[l].view(-1, C)
                        spike_buffers[l] = spike_buffers[l].view(-1, C)

                #print('layer_corr: {}'.format(layer_corr.shape))
                #for l in range(len(hooks)+1):
                for l in range(len(hooks)):
                    i = 0
                    for b in range(batch_size):
                        corr_layer = np.corrcoef(buffers[l][b].cpu(), spike_buffers[l][b].cpu())
                        layer_corr[l][batch_num*batch_size+b] = corr_layer[0,1]
                        i = i+1


            '''
            ############### calling the partial non-spiking model here ##############
            #split_layer_num = 16
            split_layer_num = 14

            model_partial = create_partial_model(split_layer_num, model)
            input_size = spike_buffers[7][0].shape[1:]
            #print (input_size)
            #summary(model_partial.to('cuda'), input_size=input_size)

            #### performing inference on this partial model
            #print(spike_buffers[int(split_layer_num/2)-1][1])
            temp = torch.from_numpy(spike_buffers[int(split_layer_num/2)-1][0]).float()
            model_partial = model_partial.to(device)
            output_partial = model_partial(temp.to(device))
            #print (output_partial.size())
            _, predicted_partial = torch.max(output_partial.data, 1)

            # this is for layer 0
            # save the correlation coefficients between pixels and input spikes
            B, C, H, W = sum_in_spikes.size()
            for b in range(batch_size):
                sum_in_spikes = np.reshape(sum_in_spikes, (batch_size, C*H*W))
                in_pixels = np.reshape(images, (batch_size, C*H*W))

            for b in range(batch_size):
                corr_in = np.corrcoef(in_pixels[b], sum_in_spikes[b] / time_window * max_acts[0])
                image_corr[i*batch_size+b] = corr_in[0,1]

            '''


            # accumulating output spikes for all images in a batch
            total_spikes_b_c = torch.zeros((batch_size, num_classes))
            for b in range(batch_size):
                total_spikes_per_input = torch.zeros((num_classes))
                for t in range(time_window):
                    total_spikes_per_input += out_spikes_t_b_c[t,b,:]
                print ("total spikes per output: {}".format(total_spikes_per_input ))
                total_spikes_b_c[b,:] = total_spikes_per_input
                #total_spikes_b_c[b,:] = total_spikes_per_input / time_window # note the change

            #_, predicted = torch.max(class_scores, 1)
            _, predicted = torch.max(total_spikes_b_c.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            expected_correct += (predicted_org.cpu() == labels).sum().item()
            # # correct classifications for combined ann+snn model
            #combined_model_correct += (predicted_partial.cpu() == labels).sum().item()
            predicted_partial = -1

            print('snn: {}\tann: {}\tpart: {}\treal: {}'.format(predicted, predicted_org, predicted_partial, labels))
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

            batch_num += 1

    #print(layer_corr)
    if spike_config.getboolean('save_activations'):
        np.save(os.path.join(out_dir, 'layerwise_corr'), layer_corr)

    model_accuracy = total_correct / total_images * 100
    expected_acc = expected_correct / total_images * 100
    #combined_model_acc = combined_model_correct / total_images * 100
    print('Model accuracy on {} test images: {}%\t\t\tacc. of ANN: {}%'.format(total_images, model_accuracy, expected_acc))
    #print ('Combined model accuracy: {}%'.format(combined_model_acc))
    #print('Model accuracy on {0} test images: {1:.2f}%\tacc. of ANN: {3:.4f}%'.format(total_images, model_accuracy, expected_acc))


from models.spiking_activations import SpikeRelu, spikeRelu
def create_spike_buffers(net, img_size, device='cuda:0'):
    relus = []
    for m in net.modules():
        if isinstance(m, spikeRelu):
            relus.append(m)

    hooks = [Hook(layer) for layer in relus]
    mats = create_mats(net, img_size, hooks, device)

    return hooks, mats

def create_buffers(net, img_size, device='cuda:0'):
    name_to_type = serialize_model(net)
    key_list = list(name_to_type.keys())
    relus = []
    for i in range(len(key_list)):
        if i < len(key_list)-1 and \
            type(name_to_type[key_list[i]]) == nn.Conv2d and \
                type(name_to_type[key_list[i+1]] == nn.ReLU):
            relus.append(name_to_type[key_list[i+1]])

        elif i < len(key_list)-1 and \
            type(name_to_type[key_list[i]]) == nn.Linear and \
                type(name_to_type[key_list[i+1]] == nn.ReLU):
            relus.append(name_to_type[key_list[i+1]])

        elif type(name_to_type[key_list[i]]) == nn.Linear or \
                type(name_to_type[key_list[i]]) == nn.AvgPool2d:
            relus.append(name_to_type[key_list[i]])

    hooks = [Hook(layer) for layer in relus]
    mats = create_mats(net, img_size, hooks, device)

    return hooks, mats


def create_mats(net, img_size, hooks, device='cuda:0'):
    img_size[0] = 1
    outputs = net(torch.zeros(img_size).to(device))

    mats = []
    mats.append(torch.zeros(img_size).to(device))
    for h in hooks:
        shape = h.output.size()
        if len(shape) > 2:
            curr_shape = [1, shape[1], shape[2], shape[3]]
        else:
            curr_shape = [1, shape[1]]
        #print(curr_shape)
        temp = torch.zeros(curr_shape).to(device)
        mats.append(temp)

    return mats


import ast
def plot_correlations(corr, out_dir, config):
    num_layers,num_samples = corr.shape
    num_imgs = config.getint('num_imgs')
    if num_imgs < 0:
        num_imgs = num_samples

    layer_nums = ast.literal_eval(config['layer_nums'])
    layers = [int(i) for i in layer_nums]
    plot_layers = []
    for i in layers:
        if i <= num_layers:
            plot_layers.append(i)

    plt.figure()
    #Plot correlations
    for l in plot_layers:
        #plt.plot(corr[l])
        plt.plot(corr[l][0:num_imgs])
    leg = [str(i) for i in plot_layers]

    # Plot correlations
    #for l in range(num_layers):
    #    plt.plot(corr[l])

    #leg = [str(i) for i in range(num_layers)]

    plt.legend(leg)
    plt.title('Correlation between ANN and SNN activations')
    print('Plotting correlations..')
    plt.savefig(os.path.join(out_dir, 'correlation.png'), bbox_inches='tight')


def plot_mean_var(net, spike_net, out_dir):
    "wt, bias mean and var of the original model"

    wt_mean, wt_var = [], []
    bias_mean, bias_var = [], []
    layer_num = []
    i = 1
    for m in net.modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            with torch.no_grad():
                layer_num.append(i)
                wt_mean.append(m.weight.mean().cpu().numpy())
                wt_var.append(m.weight.var().cpu().numpy())
                if m.bias is not None:
                    bias_mean.append(m.bias.mean().cpu().numpy())
                    bias_var.append(m.bias.var().cpu().numpy())
                i += 1

    "wt, bias mean and var of the spiking model"
    wt_mean_s, wt_var_s = [], []
    bias_mean_s, bias_var_s = [], []
    layer_num_s = []
    i = 1
    for m in spike_net.modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            with torch.no_grad():
                layer_num_s.append(i)
                wt_mean_s.append(m.weight.mean().cpu().numpy())
                wt_var_s.append(m.weight.var().cpu().numpy())
                if m.bias is not None:
                    bias_mean_s.append(m.bias.mean().cpu().numpy())
                    bias_var_s.append(m.bias.var().cpu().numpy())
                i += 1

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(layer_num, wt_mean, 'ro', label='mean')
    plt.plot(layer_num, wt_var, 'c^', label='variance')
    plt.title('original model weights')
    plt.legend()

    plt.subplot(2, 2, 2)
    if len(bias_mean) > 0:
        plt.plot(layer_num, bias_mean, 'go', label='mean')
        plt.plot(layer_num, bias_var, 'b^', label='variance')
        plt.title('original model biases')
        plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(layer_num, wt_mean_s, 'ro', label='mean')
    plt.plot(layer_num, wt_var_s, 'c^', label='variance')
    plt.title('spike model weights')
    plt.legend()
    plt.xlabel('layer number')

    plt.subplot(2, 2, 4)
    if len(bias_mean_s) > 0:
        plt.plot(layer_num, bias_mean_s, 'go', label='mean')
        plt.plot(layer_num, bias_var_s, 'b^', label='variance')
        plt.title('spike model biases')
        plt.legend()
        plt.xlabel('layer number')

    plt.savefig(os.path.join(out_dir, 'mean_var.png'), bbox_inches='tight')



# rand_val corresponds to vmem
# in_val/max_in_val corresponds to threshold
def condition(rand_val, in_val, abs_max_val):
    if rand_val <= (abs(in_val) / abs_max_val):
        return (np.sign(in_val))
    else:
        return 0

def poisson_spikes(pixel_vals):
    out_spikes = np.zeros(pixel_vals.shape)
    for b in range(pixel_vals.shape[0]):
        random_inputs = np.random.rand(pixel_vals.shape[1],pixel_vals.shape[2], pixel_vals.shape[3])
        single_img = pixel_vals[b,:,:,:]
        #max_val = np.amax(single_img) # note: shouldn't this be max(abs(single_img)) ??
        max_val = np.amax(abs(single_img)) # note: shouldn't this be max(abs(single_img)) ??
        vfunc = np.vectorize(condition)
        out_spikes[b,:,:,:] = vfunc(random_inputs, single_img, max_val)
    return out_spikes

def sanity_check(net, spike_net, max_acts):

    num = 0
    num_to_type = {}
    for name, module in net.named_modules():
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            num_to_type[num] = module
            num += 1

    num = 0
    i = 0
    for name, module in spike_net.named_modules():
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            if not(torch.all(module.weight.data == num_to_type[num].weight.data)):
                print ('weights dont match at layer {}'.format(num))

            if module.bias is not None:
                if not(torch.all(module.bias.data == num_to_type[num].bias.data / max_acts[i])):
                    print ('biases dont match at layer {}'.format(num))

            num += 1
            i += 1


from models import mobilenet_spiking, vgg16_spiking
def createSpikingModel(net, arch, num_classes, spike_config, thresholds, max_acts, device='cuda:0'):
    ''' check if model has BN layers '''
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            print('model {} has BN layers. Can\'t spikify. Exiting...'.format(arch))
            exit()

    clamp_slope = spike_config.getint('clamp_slope')
    reset = spike_config['reset']

    spike_net =  None
    if 'mobilenet_cifar10' in arch:
        spike_net = mobilenet_spiking.mobilenet_spike(thresholds, clamp_slope, num_classes, device, reset)
    if 'vgg_cifar10' in arch:
        spike_net = vgg16_spiking.vgg16_spike(thresholds, clamp_slope, num_classes, device, reset)

    num = 0
    num_to_type = {}
    for name, module in net.named_modules():
        #print(name)
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            num_to_type[num] = module
            num += 1

    num = 0
    for name, module in spike_net.named_modules():
        if type(module) == nn.Conv2d or type(module) == nn.Linear:
            module.weight.data = num_to_type[num].weight.data.clone()
            if num_to_type[num].bias is not None and module.bias is not None:
                module.bias.data = num_to_type[num].bias.data.clone()
            num += 1


    # adjust weights
    i = 0
    for layer in spike_net.modules():
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear: # note the addition of linear layer to this
            if layer.bias is not None:
                #temp = layer.bias / max_acts[i+1] # when all vth is 1
                temp = layer.bias / max_acts[i]
                layer.bias = torch.nn.Parameter(temp)
            i += 1


    '''
    # when all vth is 1
    j = 0
    for layer in new_model.modules():
        if type(layer) == torch.nn.Conv2d or type(layer) == nn.Linear:
            temp = max_acts[j] / max_acts[j+1]
            layer.weight = torch.nn.Parameter(layer.weight * temp)
            j += 1
    '''

    return spike_net


def compute_thresholds(net, dataloader, out_dir, percentile=99.9, device='cuda:0'):
    name_to_type = serialize_model(net)
    key_list = list(name_to_type.keys())
    relus = []
    for i in range(len(key_list)):
        if i < len(key_list)-1 and \
            type(name_to_type[key_list[i]]) == nn.Conv2d and \
                type(name_to_type[key_list[i+1]] == nn.ReLU):
            relus.append(name_to_type[key_list[i]])

        elif i < len(key_list)-1 and \
            type(name_to_type[key_list[i]]) == nn.Linear and \
                type(name_to_type[key_list[i+1]] == nn.ReLU):
            relus.append(name_to_type[key_list[i]])

        elif type(name_to_type[key_list[i]]) == nn.Linear or \
                type(name_to_type[key_list[i]]) == nn.AvgPool2d:
            relus.append(name_to_type[key_list[i]])

    hooks = [Hook(layer) for layer in relus]

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    acts = np.zeros((len(hooks)+1, 10000))
    with torch.no_grad():
        for n, data in enumerate(dataloader):
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            outputs = net(images)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_idx = n
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            batch_size = targets.size(0)
            img_max = np.amax(images.cpu().numpy(), axis=(1,2,3))
            acts[0,n*batch_size:(n+1)*batch_size] = img_max
            for i, hook in enumerate(hooks):
                if len(hook.output.size()) > 2:
                    acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(), axis=(1,2,3))
                else:
                    acts[i+1][n*batch_size:(n+1)*batch_size] = np.amax(hook.output.cpu().numpy(), axis=1)

    max_val = np.percentile(acts, percentile, axis=1)
    print('max activations: ', max_val)
    thresholds = torch.zeros(len(max_val)-1)
    for i in range(len(thresholds)):
        thresholds[i] = max_val[i+1] / max_val[i]
    np.savetxt(os.path.join(out_dir, 'thresholds.txt'), thresholds, fmt='%.5f')
    np.savetxt(os.path.join(out_dir, 'max_acts.txt'), max_val, fmt='%.5f')
    print('thresholds: ', thresholds)


class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()
