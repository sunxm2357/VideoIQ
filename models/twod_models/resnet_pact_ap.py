import sys
sys.path.insert(0, '../../')
from functools import partial
from inspect import signature
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.twod_models.common import TemporalPooling
from models.twod_models.fpn import FPN
from models.twod_models.temporal_modeling import temporal_modeling_module
from models.inflate_from_2d_model import convert_rgb_model_to_others
from models.twod_models.ops.imagenet_pact import activation_quantize_fn2, conv2d_Q_fn_dorefa2

__all__ = ['resnet_pact_ap', 'ResNet_PACT_AP']
DEBUG = False

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def one_hot(array, labels):
    num_classes = len(labels)
    device_id = array.get_device()
    array_flatten = array.contiguous().view(-1)
    device_id = device_id if device_id >= 0 else 'cpu'
    array_index = torch.tensor([labels.index(a) for a in array_flatten], device=device_id).contiguous().view(-1, 1)
    y_onehot_flatten = torch.zeros(array_flatten.shape[0], num_classes, device=device_id)

    # In your for loop
    y_onehot_flatten.scatter_(1, array_index, 1)
    y_onehot = y_onehot_flatten.view(*array.shape, -1)
    return y_onehot


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def conv3x3(in_planes, out_planes, mean_aligned=False,  stride=1):
    """3x3 convolution with padding"""
    Conv2d = conv2d_Q_fn_dorefa2(mean_aligned)
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


def conv1x1(in_planes, out_planes, mean_aligned=False, stride=1):
    """1x1 convolution"""
    Conv2d = conv2d_Q_fn_dorefa2(mean_aligned)
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3_fp(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


def conv1x1_fp(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_frames, w_bit_width_family, a_bit_width_family, q_init=[10],
                 mean_aligned=False, switch_bn=False, switch_clipval=False, quantize_fp=True, stride=1, downsamples=None, temporal_module=None):
        if len(q_init) == 1:
            q_init = q_init * len(a_bit_width_family)
        super(BasicBlock, self).__init__()
        self.w_bit_width_family = w_bit_width_family
        self.a_bit_width_family = a_bit_width_family
        self.w_max_bit_width = max(w_bit_width_family)
        self.switch_bn = switch_bn
        self.switch_clipval = switch_clipval
        self.quantize_fp = quantize_fp
        if self.switch_clipval:
            for a_idx, a_bit in enumerate(self.a_bit_width_family):
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
                    self.register_parameter('q_alpha1_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
                    self.register_parameter('q_alpha2_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
        else:
            for a_bit in self.a_bit_width_family:
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
            if not (len(a_bit_width_family) == 1 and a_bit_width_family[0] == 32 and not quantize_fp):
                self.register_parameter('q_alpha1', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha2', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))

        if len(w_bit_width_family) == 1 and w_bit_width_family[0] == 32 and not quantize_fp:
            self.conv1 = conv3x3_fp(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, mean_aligned, stride)
        if self.switch_bn:
            for w_bit in self.w_bit_width_family:
                if (not quantize_fp) and w_bit == 32:
                    setattr(self, 'bn1', nn.BatchNorm2d(planes))
                else:
                    setattr(self, 'bn1_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        if len(w_bit_width_family) == 1 and w_bit_width_family[0] == 32 and not quantize_fp:
            self.conv2 = conv3x3_fp(planes, planes)
        else:
            self.conv2 = conv3x3(planes, planes, mean_aligned)

        if self.switch_bn:
            for w_bit in self.w_bit_width_family:
                if (not quantize_fp) and w_bit == 32:
                    setattr(self, 'bn2', nn.BatchNorm2d(planes))
                else:
                    setattr(self, 'bn2_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        if downsamples is not None:
            assert len(downsamples) == len(self.w_bit_width_family)
            for w_idx, w_bit in enumerate(self.w_bit_width_family):
                setattr(self, 'downsample_%d' % w_bit, downsamples[w_idx])
            self.downsample = True
        else:
            self.downsample = False

        self.stride = stride

        if temporal_module is not None:
            self.tam = temporal_module(duration=num_frames, channels=inplanes)
            # TODO: to change
            if self.switch_clipval:
                for a_idx, a_bit in enumerate(self.a_bit_width_family):
                    if a_bit != 32:
                        self.register_parameter('q_alpha_tam_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1),  requires_grad=True))
            else:
                if not (len(a_bit_width_family) == 1 and a_bit_width_family[0] == 32):
                    self.register_parameter('q_alpha_tam', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
        else:
            self.tam = None

    def forward(self, x, w_bit, a_bit):
        if a_bit != 32 or self.quantize_fp:
            if DEBUG:
                print('Quant')
            act_q1 = getattr(self, 'act_q%d' % a_bit)
            if self.switch_clipval:
                q_alpha1 = getattr(self, 'q_alpha1_%d' % a_bit)
            else:
                q_alpha1 = self.q_alpha1
            x = act_q1(x, q_alpha1)
        else:
            if DEBUG:
                print('Not Quant')

        identity = x
        if self.tam is not None:
            x = self.tam(x)
            if a_bit != 32 or self.quantize_fp:
                if DEBUG:
                    print('Quant')
                act_q_tam = getattr(self, 'act_q%d' % a_bit)
                if self.switch_clipval:
                    q_alpha_tam = getattr(self, 'q_alpha_tam_%d' % a_bit)
                else:
                    q_alpha_tam = self.q_alpha_tam
                x = act_q_tam(x, q_alpha_tam)
            else:
                if DEBUG:
                    print('Not Quant')

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv1(x)
            if DEBUG:
                print('Not Quant')
        else:
            out = self.conv1(x, self.w_max_bit_width, w_bit)
            if DEBUG:
                print('Quant')

        if self.switch_bn:
            if w_bit != 32 or self.quantize_fp:
                bn1 = getattr(self, 'bn1_%d' % w_bit)
                if DEBUG:
                    print('Quant')
            else:
                bn1 = self.bn1
                if DEBUG:
                    print('Not Quant')
        else:
            bn1 = self.bn1
        out = bn1(out)
        out = self.relu(out)

        if a_bit != 32 or self.quantize_fp:
            if DEBUG:
                print('Quant')
            act_q2 = getattr(self, 'act_q%d' % a_bit)
            if self.switch_clipval:
                q_alpha2 = getattr(self, 'q_alpha2_%d' % a_bit)
            else:
                q_alpha2 = self.q_alpha2
            out = act_q2(out, q_alpha2)
        else:
            if DEBUG:
                print('Not Quant')

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv2(out)
            if DEBUG:
                print('Not Quant')
        else:
            out = self.conv2(out, self.w_max_bit_width, w_bit)
            if DEBUG:
                print('Quant')

        if self.switch_bn:
            if w_bit != 32 or self.quantize_fp:
                bn2 = getattr(self, 'bn2_%d' % w_bit)
                if DEBUG:
                    print('Quant')
            else:
                bn2 = self.bn2
                if DEBUG:
                    print('Not Quant')
        else:
            bn2 = self.bn2
        out = bn2(out)

        if self.downsample:
            if self.tam is not None:
                if 'TSM' in self.tam.name():
                    downsample = getattr(self, 'downsample_%d' % w_bit)
                    if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(identity)
                        if DEBUG:
                            print('Not Quant')
                    else:
                        identity = downsample(identity, self.w_max_bit_width, w_bit)
                        if DEBUG:
                            print('Quant')
                else:
                    downsample = getattr(self, 'downsample_%d' % w_bit)
                    if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(x)
                        if DEBUG:
                            print('Not Quant')
                    else:
                        identity = downsample(x, self.w_max_bit_width, w_bit)
                        if DEBUG:
                            print('Quant')
            else:
                downsample = getattr(self, 'downsample_%d' % w_bit)
                if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
                    identity = downsample(x)
                    if DEBUG:
                        print('Not Quant')
                else:
                    identity = downsample(x, self.w_max_bit_width, w_bit)
                    if DEBUG:
                        print('Quant')

        out += identity
        out = self.relu(out)

        return out, w_bit, a_bit


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_frames,  w_bit_width_family,  a_bit_width_family, q_init=[10],
                 mean_aligned=False, switch_bn=False, switch_clipval=False, quantize_fp=True, stride=1, downsamples=None, temporal_module=None):
        super(Bottleneck, self).__init__()
        if len(q_init) == 1:
            q_init = q_init * len(a_bit_width_family)
        self.w_bit_width_family = w_bit_width_family
        self.a_bit_width_family = a_bit_width_family
        self.w_max_bit_width = max(w_bit_width_family)
        self.switch_bn = switch_bn
        self.switch_clipval = switch_clipval
        self.quantize_fp = quantize_fp
        if self.switch_clipval:
            for a_idx, a_bit in enumerate(self.a_bit_width_family):
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
                    self.register_parameter('q_alpha1_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
                    self.register_parameter('q_alpha2_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
                    self.register_parameter('q_alpha3_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
        else:
            for a_bit in self.a_bit_width_family:
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
            if not (len(a_bit_width_family) == 1 and a_bit_width_family[0] == 32 and not quantize_fp):
                self.register_parameter('q_alpha1', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha2', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha3', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not quantize_fp:
            self.conv1 = conv1x1_fp(inplanes, planes)
        else:
            self.conv1 = conv1x1(inplanes, planes, mean_aligned)
        if self.switch_bn:
            for w_bit in self.w_bit_width_family:
                if w_bit == 32 and not quantize_fp:
                    setattr(self, 'bn1', nn.BatchNorm2d(planes))
                else:
                    setattr(self, 'bn1_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn1 = nn.BatchNorm2d(planes)

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not quantize_fp:
            self.conv2 = conv3x3_fp(planes, planes, stride)
        else:
            self.conv2 = conv3x3(planes, planes,mean_aligned, stride)
        if self.switch_bn:
            for w_bit in self.w_bit_width_family:
                if w_bit == 32 and not quantize_fp:
                    self.bn2 = nn.BatchNorm2d(planes)
                else:
                    setattr(self, 'bn2_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not quantize_fp:
            self.conv3 = conv1x1_fp(planes, planes * self.expansion)
        else:
            self.conv3 = conv1x1(planes, planes * self.expansion, mean_aligned)

        if self.switch_bn:
            for w_bit in self.w_bit_width_family:
                if w_bit == 32 and not quantize_fp:
                    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                else:
                    setattr(self, 'bn3_%d' % w_bit, nn.BatchNorm2d(planes * self.expansion))
        else:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()

        if downsamples is not None:
            assert len(downsamples) == len(self.w_bit_width_family)
            for w_idx, w_bit in enumerate(self.w_bit_width_family):
                setattr(self, 'downsample_%d' % w_bit, downsamples[w_idx])
            self.downsample = True
        else:
            self.downsample = False

        self.stride = stride

        if temporal_module is not None:
            self.tam = temporal_module(duration=num_frames, channels=inplanes)
            # TODO: to change
            if self.switch_clipval:
                for a_idx, a_bit in enumerate(self.a_bit_width_family):
                    if a_bit != 32 or quantize_fp:
                        self.register_parameter('q_alpha_tam_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1),
                                                                                   requires_grad=True))
            else:
                if not (len(a_bit_width_family) == 1 and a_bit_width_family[0] == 32 and not quantize_fp):
                    self.register_parameter('q_alpha_tam', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
        else:
            self.tam = None

    def forward(self, x, w_bit, a_bit):
        if a_bit != 32 or self.quantize_fp:
            act_q1 = getattr(self, 'act_q%d' % a_bit)
            if self.switch_clipval:
                q_alpha1 = getattr(self, 'q_alpha1_%d' % a_bit)
            else:
                q_alpha1 = self.q_alpha1
            x = act_q1(x, q_alpha1)

        identity = x
        if self.tam is not None:
            x = self.tam(x)
            if a_bit != 32 or self.quantize_fp:
                act_q_tam = getattr(self, 'act_q%d' % a_bit)
                if self.switch_clipval:
                    q_alpha_tam = getattr(self, 'q_alpha_tam_%d' % a_bit)
                else:
                    q_alpha_tam = self.q_alpha_tam
                x = act_q_tam(x, q_alpha_tam)

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv1(x)
        else:
            out = self.conv1(x, self.w_max_bit_width, w_bit)

        if self.switch_bn:
            if w_bit != 32 or self.quantize_fp:
                bn1 = getattr(self, 'bn1_%d' % w_bit)
            else:
                bn1 = self.bn1
        else:
            bn1 = self.bn1
        out = bn1(out)
        out = self.relu(out)

        if a_bit != 32 or self.quantize_fp:
            act_q2 = getattr(self, 'act_q%d' % a_bit)
            if self.switch_clipval:
                q_alpha2 = getattr(self, 'q_alpha2_%d' % a_bit)
            else:
                q_alpha2 = self.q_alpha2
            out = act_q2(out, q_alpha2)

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv2(out)
        else:
            out = self.conv2(out, self.w_max_bit_width, w_bit)

        if self.switch_bn:
            if w_bit != 32 or self.quantize_fp:
                bn2 = getattr(self, 'bn2_%d' % w_bit)
            else:
                bn2 = self.bn2
        else:
            bn2 = self.bn2
        out = bn2(out)
        out = self.relu(out)

        if a_bit != 32 or self.quantize_fp:
            act_q3 = getattr(self, 'act_q%d' % a_bit)
            if self.switch_clipval:
                q_alpha3 = getattr(self, 'q_alpha3_%d' % a_bit)
            else:
                q_alpha3 = self.q_alpha3
            out = act_q3(out, q_alpha3)

        if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv3(out)
        else:
            out = self.conv3(out, self.w_max_bit_width, w_bit)

        if self.switch_bn:
            if w_bit != 32 or self.quantize_fp:
                bn3 = getattr(self, 'bn3_%d' % w_bit)
            else:
                bn3 = self.bn3
        else:
            bn3 = self.bn3
        out = bn3(out)

        if self.downsample:
            if self.tam is not None:
                if 'TSM' in self.tam.name():
                    downsample = getattr(self, 'downsample_%d' % w_bit)
                    if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(identity)
                    else:
                        identity = downsample(identity, self.w_max_bit_width, w_bit)
                else:
                    downsample = getattr(self, 'downsample_%d' % w_bit)
                    if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(x)
                    else:
                        identity = downsample(x, self.w_max_bit_width, w_bit)
            else:
                downsample = getattr(self, 'downsample_%d' % w_bit)
                if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not self.quantize_fp:
                    identity = downsample(x)
                else:
                    identity = downsample(x, self.w_max_bit_width, w_bit)

        out += identity
        out = self.relu(out)

        return out, w_bit, a_bit


class ResNet_PACT_AP(nn.Module):
    def __init__(self, depth, num_frames, w_bit_width_family, a_bit_width_family, q_init, mean_aligned=False,
                 switch_bn=False, switch_clipval=False, is_32fp=False,
                 num_classes=1000, dropout=0.5, zero_init_residual=False,
                 without_t_stride=False, temporal_module=None, fpn_dim=-1, pooling_method='max',
                 input_channels=3, tam_pos='all'):
        super(ResNet_PACT_AP, self).__init__()
        self.w_bit_width_family = w_bit_width_family
        self.a_bit_width_family = a_bit_width_family
        self.mean_aligned = mean_aligned
        self.switch_bn = switch_bn
        self.switch_clipval = switch_clipval
        self.is_32fp = is_32fp
        self.w_bit_list = "".join([str(bit) for bit in self.w_bit_width_family])
        self.a_bit_list = "".join([str(bit) for bit in self.a_bit_width_family])
        self.q_init = q_init
        uniques = np.unique(self.q_init)
        if len(uniques) == 1:
            self.q_init_list = str(self.q_init[0])
        else:
            self.q_init_list = "".join([str(bit) for bit in self.q_init])
        self.pooling_method = pooling_method.lower()
        block = BasicBlock if depth < 50 else Bottleneck
        layers = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]}[depth]

        self.num_blocks = {18: 16, 34: 32, 50: 48, 101: 99, 152: 150}
        self.num_layers = {18: 8, 34: 16, 50: 16, 101: 33, 152: 50}
        self.depth = depth
        self.register_buffer('swap_r', torch.rand(self.num_layers[self.depth]))

        self.temporal_module = temporal_module
        self.num_frames = num_frames
        self.orig_num_frames = num_frames
        self.num_classes = num_classes
        self.without_t_stride = without_t_stride
        self.fpn_dim = fpn_dim
        # TODO: only for ResNet-18
        self.tam_pos = tam_pos if self.depth == 18 else 'all'

        self.inplanes = 64
        # First and Last layers are full-precision
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       skip_tam=True if self.tam_pos == 'last' else False, quantize_fp=True)
        if not self.without_t_stride:
            self.pool1 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       skip_tam=True if self.tam_pos == 'last' else False, quantize_fp=True)
        if not self.without_t_stride:
            self.pool2 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       skip_tam=True if self.tam_pos == 'first' else False, quantize_fp=True)
        if not self.without_t_stride:
            self.pool3 = TemporalPooling(self.num_frames, 3, 2, self.pooling_method)
            self.num_frames = self.num_frames // 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       skip_tam=True if self.tam_pos == 'first' else False, quantize_fp=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        if self.fpn_dim > 0:
            self.fpn = FPN(self.fpn_dim)
            self.fc2 = nn.Linear(fpn_dim, num_classes)
            self.fc3 = nn.Linear(fpn_dim, num_classes)
            self.fc4 = nn.Linear(fpn_dim, num_classes)
            self.fc5 = nn.Linear(fpn_dim, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.is_32fp:
            self.inplanes = 64
            self.switch_bn, self.switch_clipval = False, False
            self.w_bit_width_family = [32]
            self.a_bit_width_family = [32]
            self.conv1_32fp = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1_32fp = nn.BatchNorm2d(64)
            self.layer1_32fp = self._make_layer(block, 64, layers[0],
                                                skip_tam=True if self.tam_pos == 'last' else False, quantize_fp=False)
            self.layer2_32fp = self._make_layer(block, 128, layers[1], stride=2,
                                                skip_tam=True if self.tam_pos == 'last' else False, quantize_fp=False)
            self.layer3_32fp = self._make_layer(block, 256, layers[2], stride=2,
                                                skip_tam=True if self.tam_pos == 'first' else False, quantize_fp=False)
            self.layer4_32fp = self._make_layer(block, 512, layers[3], stride=2,
                                                skip_tam=True if self.tam_pos == 'first' else False, quantize_fp=False)
            self.fc_32fp = nn.Linear(512 * block.expansion, num_classes)
            self.switch_bn, self.switch_clipval = switch_bn, switch_clipval
            self.w_bit_width_family = w_bit_width_family
            self.a_bit_width_family = a_bit_width_family

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.softmax = nn.Softmax(dim=0)

        # freeze the full precision model
        for name, param in self.named_parameters():
            if '32fp' in name:
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, skip_tam=False, quantize_fp=True):
        downsamples = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsamples = []
            if len(self.w_bit_width_family) == 1 and self.w_bit_width_family[0] == 32 and not quantize_fp:
                conv_layer = conv1x1_fp(self.inplanes, planes * block.expansion, stride)
            else:
                conv_layer = conv1x1(self.inplanes, planes * block.expansion, self.mean_aligned, stride)

            if self.switch_bn:
                for _ in self.w_bit_width_family:
                    downsamples.append(mySequential(conv_layer,
                    nn.BatchNorm2d(planes * block.expansion),
                    ))
            else:
                bn = nn.BatchNorm2d(planes * block.expansion)
                for _ in self.w_bit_width_family:
                    downsamples.append(mySequential(conv_layer, bn))

        if self.tam_pos == 'half':
            skip_tam_1 = False
            skip_tam_2 = True
        elif self.tam_pos == 'half_2':
            skip_tam_1 = True
            skip_tam_2 = False
        else:
            skip_tam_1 = skip_tam
            skip_tam_2 = skip_tam


        layers = []
        block_tmp = block(self.inplanes, planes, self.num_frames, self.w_bit_width_family, self.a_bit_width_family,
                          self.q_init, self.mean_aligned, self.switch_bn, self.switch_clipval, quantize_fp, stride,
                          downsamples, temporal_module=self.temporal_module if not skip_tam_1 else None)
        layers.append(block_tmp)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            block_tmp = block(self.inplanes, planes, self.num_frames, self.w_bit_width_family, self.a_bit_width_family,
                              self.q_init,  self.mean_aligned, self.switch_bn, self.switch_clipval, quantize_fp,
                              temporal_module=self.temporal_module if not skip_tam_2 else None)
            layers.append(block_tmp)

        return mySequential(*layers)

    def forward_32fp(self, x, batch_size, w_bit, a_bit):
        x = self.conv1_32fp(x)
        x = self.bn1_32fp(x)
        x = self.relu(x)
        fp1 = self.maxpool(x)

        fp2, w_bit, a_bit = self.layer1_32fp(fp1, w_bit, a_bit)
        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3, w_bit, a_bit = self.layer2_32fp(fp2_d, w_bit, a_bit)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4, w_bit, a_bit = self.layer3_32fp(fp3_d, w_bit, a_bit)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5, w_bit, a_bit = self.layer4_32fp(fp4_d, w_bit, a_bit)

        x_32fp = self.avgpool(fp5)
        x_32fp = x_32fp.view(x_32fp.size(0), -1)
        x_32fp = self.dropout(x_32fp)
        x_32fp = self.fc_32fp(x_32fp)

        n_t, c = x_32fp.shape
        out_32fp = x_32fp.view(batch_size, -1, c)

        # average the prediction from all frames
        out_32fp = torch.mean(out_32fp, dim=1)
        return out_32fp

    def forward(self, x, w_bit=4, a_bit=4, **kwargs):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)

        if self.is_32fp and max(self.w_bit_width_family) == w_bit:
            w_bits_32fp = 32
            a_bits_32fp = 32

            with torch.no_grad():
                x_32fp = x.clone()
                if DEBUG:
                    print('=============Start 32fp==============')
                out_32fp = self.forward_32fp(x_32fp, batch_size, w_bits_32fp, a_bits_32fp)
        else:
            out_32fp = None

        if DEBUG:
            print('=============Start Normal Forward==============')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        fp1 = self.maxpool(x)
        fp2, w_bit, a_bit = self.layer1(fp1, w_bit, a_bit)

        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3, w_bit, a_bit = self.layer2(fp2_d, w_bit, a_bit)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4, w_bit, a_bit = self.layer3(fp3_d, w_bit, a_bit)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5, w_bit, a_bit = self.layer4(fp4_d, w_bit, a_bit)

        if self.fpn_dim > 0:
            fp2, fp3, fp4, fp5 = self.fpn(fp2, fp3, fp4, fp5, batch_size)
            pred2 = torch.mean(self.fc2(fp2).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)
            pred3 = torch.mean(self.fc3(fp3).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)
            pred4 = torch.mean(self.fc4(fp4).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)
            pred5 = torch.mean(self.fc5(fp5).view(batch_size, -1, self.num_classes),
                               dim=1, keepdim=True)

            out = torch.cat((pred2, pred3, pred4, pred5), dim=1)
        else:
            x = self.avgpool(fp5)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)

            n_t, c = x.shape
            out = x.view(batch_size, -1, c)

        # average the prediction from all frames
        out = torch.mean(out, dim=1)
        return out, out_32fp

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = ''
        if self.temporal_module is not None:
            param = signature(self.temporal_module).parameters
            temporal_module = str(param['name']).split("=")[-1][1:-1]
            blending_frames = str(param['blending_frames']).split("=")[-1]
            blending_method = str(param['blending_method']).split("=")[-1][1:-1]
            dw_conv = True if str(param['dw_conv']).split("=")[-1] == 'True' else False
            name += "{}-b{}-{}{}-".format(temporal_module, blending_frames,
                                         blending_method,
                                         "" if dw_conv else "-allc")
        name += 'resnet_pact_ap-{}-w{}a{}init{}'.format(self.depth, self.w_bit_list, self.a_bit_list, self.q_init_list)
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)
        if self.fpn_dim > 0:
            name += "-fpn{}".format(self.fpn_dim)

        return name

    @property
    def num_layer(self):
        return self.num_layers[self.depth]

def make_non_local_v2(net, num_frames):
    from models.twod_models.ops.non_local import NL3DWrapperV2
    if isinstance(net.layer1[0], Bottleneck):
        net.layer2 = nn.Sequential(
            net.layer2[0],
            NL3DWrapperV2(net.layer2[0].conv3.out_channels, num_frames),
            net.layer2[1],
            net.layer2[2],
            NL3DWrapperV2(net.layer2[2].conv3.out_channels, num_frames),
            net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            net.layer3[0],
            NL3DWrapperV2(net.layer3[0].conv3.out_channels, num_frames),
            net.layer3[1],
            net.layer3[2],
            NL3DWrapperV2(net.layer3[2].conv3.out_channels, num_frames),
            net.layer3[3],
            net.layer3[4],
            NL3DWrapperV2(net.layer3[4].conv3.out_channels, num_frames),
            net.layer3[5],
        )
    else:
        net.layer2 = nn.Sequential(
            net.layer2[0],
            NL3DWrapperV2(net.layer2[0].conv2.out_channels, num_frames),
            net.layer2[1]
        )
        net.layer3 = nn.Sequential(
            net.layer3[0],
            NL3DWrapperV2(net.layer3[0].conv2.out_channels, num_frames),
            net.layer3[1]
        )

    return net


def make_non_local(net, n_segment):
    #  only works for resnet-50
    from models.twod_models.ops.non_local import NL3DWrapper
    net.layer2 = nn.Sequential(
        NL3DWrapper(net.layer2[0], n_segment),
        net.layer2[1],
        NL3DWrapper(net.layer2[2], n_segment),
        net.layer2[3],
    )
    net.layer3 = nn.Sequential(
        NL3DWrapper(net.layer3[0], n_segment),
        net.layer3[1],
        NL3DWrapper(net.layer3[2], n_segment),
        net.layer3[3],
        NL3DWrapper(net.layer3[4], n_segment),
        net.layer3[5],
    )


def resnet_pact_ap(depth, num_classes, without_t_stride, groups, temporal_module_name,
                       dw_conv, blending_frames, blending_method, dropout, fpn_dim, pooling_method,
                       input_channels, tam_pos, q_init, w_bit_width_family, a_bit_width_family, mean_aligned,
                       switch_bn, switch_clipval, is_32fp, imagenet_pretrained=True, imagenet_path=None,
                       **kwargs):

    add_non_local = False
    temporal_module = None
    if temporal_module_name is None or temporal_module_name == 'TSN':
        temporal_module = None
    else:
        temporal_module_name = temporal_module_name.split("+")
        for x in temporal_module_name:
            if x == 'NonLocal':
                add_non_local = True
            else:
                temporal_module = partial(temporal_modeling_module, name=x,
                                          dw_conv=dw_conv, blending_frames=blending_frames,
                                          blending_method=blending_method)

    model = ResNet_PACT_AP(depth, num_frames=groups, w_bit_width_family=w_bit_width_family,
                           a_bit_width_family=a_bit_width_family,
                           q_init=q_init, mean_aligned=mean_aligned, switch_bn=switch_bn,
                           switch_clipval=switch_clipval, is_32fp=is_32fp, num_classes=num_classes,
                           without_t_stride=without_t_stride, temporal_module=temporal_module,
                           dropout=dropout, fpn_dim=fpn_dim, pooling_method=pooling_method,
                           input_channels=input_channels, tam_pos=tam_pos)

    if add_non_local:
        model = make_non_local_v2(model, groups)

    if imagenet_pretrained:
        if imagenet_path is None:
            state_dict = model_zoo.load_url(model_urls['resnet{}'.format(depth)], map_location='cpu')
            old_state_dict = model.state_dict()
            for k, v in old_state_dict.items():
                if k in state_dict.keys():
                    old_state_dict[k] = state_dict[k]
                else:
                    tokens = k.split('.')
                    new_tokens = []
                    for t in tokens:
                        if 'downsample' in t or 'bn' in t or '32fp' in t:
                            subtokens = t.split('_')
                            if len(subtokens) == 2:
                                new_tokens.append(subtokens[0])
                            else:
                                new_tokens.append(t)
                        else:
                            new_tokens.append(t)

                    new_k = '.'.join(new_tokens)
                    if new_k in state_dict.keys():
                        old_state_dict[k] = state_dict[new_k]
                    else:
                        print('Skip %s, new k is %s' % (k, new_k))

            state_dict = old_state_dict
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            if 'fc_32fp.weight' in state_dict.keys() and 'fc_32fp.bias' in state_dict.keys():
                state_dict.pop('fc_32fp.weight', None)
                state_dict.pop('fc_32fp.bias', None)
        else:
            state_dict = torch.load(imagenet_path, map_location='cpu')['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "")
                if "num_batches_tracked" not in new_k:
                    new_state_dict[k.replace("module.", "")] = v
            state_dict = new_state_dict

            old_state_dict = model.state_dict()
            for k, v in old_state_dict.items():
                if k in state_dict.keys():
                    old_state_dict[k] = state_dict[k]
                else:
                    tokens = k.split('.')
                    new_tokens = []
                    for t in tokens:
                        if 'downsample' in t or 'bn' in t or '32fp' in t:
                            subtokens = t.split('_')
                            if len(subtokens) == 2:
                                new_tokens.append(subtokens[0])
                            else:
                                new_tokens.append(t)
                        else:
                            new_tokens.append(t)

                    new_k = '.'.join(new_tokens)
                    if new_k in state_dict.keys():
                        old_state_dict[k] = state_dict[new_k]
                    else:
                        print('Skip %s, new k is %s' % (k, new_k))
            state_dict = old_state_dict

        if input_channels != 3:  # convert the RGB model to others, like flow
            state_dict = convert_rgb_model_to_others(state_dict, input_channels, 7)
        model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == '__main__':
    from torchsummary import torchsummary
    model = resnet_pact_ap(18, 200, without_t_stride=True, groups=16, dropout=0.5,
                           temporal_module_name='TSN', dw_conv=True, blending_frames=3, tam_pos='half_2',
                           blending_method='sum', fpn_dim=-1, pooling_method='max', input_channels=3,
                           w_bit_width_family=[32, 4, 2],
                           a_bit_width_family=[32, 4, 2], q_init=[2], mean_aligned=False, switch_bn=True, switch_clipval=True, is_32fp=True)

    # dummy_data = (48, 224, 224)
    # model.eval()
    # model_summary = torchsummary.summary(model, input_size=dummy_data)
    # print(model)
    # print(model_summary)
    # import pdb
    # pdb.set_trace()
    print(model.network_name)
    input_dummy = torch.ones(2, 48, 224, 224)
    output, output_32fp = model(input_dummy, w_bit=32, a_bit=32)
    import pdb
    pdb.set_trace()

