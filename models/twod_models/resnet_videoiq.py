import sys
sys.path.insert(0, '../../')
from functools import partial
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.twod_models.common import TemporalPooling
from models.twod_models.fpn import FPN
from models.twod_models.temporal_modeling import temporal_modeling_module
from models.inflate_from_2d_model import convert_rgb_model_to_others
from models.twod_models.ops.imagenet_pact import activation_quantize_fn2, conv2d_Q_fn_dorefa2
from models.twod_models.policynet import MobileNetV2
import torch.nn.functional as F

__all__ = ['resnet_videoiq', 'ResNetVideoIQ']

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

    def __init__(self, inplanes, planes, num_frames, bit_width_family, q_init=[10],
                 mean_aligned=False, switch_bn=False, switch_clipval=False, quantize_fp=True, stride=1, downsamples=None, temporal_module=None):
        if len(q_init) == 1:
            q_init = q_init * len(bit_width_family)
        super(BasicBlock, self).__init__()
        self.bit_width_family = bit_width_family
        self.max_bit_width = max(bit_width_family)
        self.switch_bn = switch_bn
        self.switch_clipval = switch_clipval
        self.quantize_fp = quantize_fp
        if self.switch_clipval:
            for a_idx, a_bit in enumerate(self.bit_width_family):
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
                    self.register_parameter('q_alpha1_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
                    self.register_parameter('q_alpha2_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
        else:
            for a_bit in self.bit_width_family:
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
            if not (len(bit_width_family) == 1 and bit_width_family[0] == 32 and not quantize_fp):
                self.register_parameter('q_alpha1', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha2', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))

        if len(bit_width_family) == 1 and bit_width_family[0] == 32 and not quantize_fp:
            self.conv1 = conv3x3_fp(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, mean_aligned, stride)
        if self.switch_bn:
            for w_bit in self.bit_width_family:
                if (not quantize_fp) and w_bit == 32:
                    setattr(self, 'bn1', nn.BatchNorm2d(planes))
                else:
                    setattr(self, 'bn1_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        if len(bit_width_family) == 1 and bit_width_family[0] == 32 and not quantize_fp:
            self.conv2 = conv3x3_fp(planes, planes)
        else:
            self.conv2 = conv3x3(planes, planes, mean_aligned)

        if self.switch_bn:
            for w_bit in self.bit_width_family:
                if (not quantize_fp) and w_bit == 32:
                    setattr(self, 'bn2', nn.BatchNorm2d(planes))
                else:
                    setattr(self, 'bn2_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        if downsamples is not None:
            assert len(downsamples) == len(self.bit_width_family)
            for w_idx, w_bit in enumerate(self.bit_width_family):
                setattr(self, 'downsample_%d' % w_bit, downsamples[w_idx])
            self.downsample = True
        else:
            self.downsample = False

        self.stride = stride

        if temporal_module is not None:
            self.tam = temporal_module(duration=num_frames, channels=inplanes)
            # TODO: to change
            if self.switch_clipval:
                for a_idx, a_bit in enumerate(self.bit_width_family):
                    if a_bit != 32:
                        self.register_parameter('q_alpha_tam_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1),  requires_grad=True))
            else:
                if not (len(bit_width_family) == 1 and bit_width_family[0] == 32):
                    self.register_parameter('q_alpha_tam', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
        else:
            self.tam = None

    def forward(self, x, bit):
        if bit != 32 or self.quantize_fp:
            act_q1 = getattr(self, 'act_q%d' % bit)
            if self.switch_clipval:
                q_alpha1 = getattr(self, 'q_alpha1_%d' % bit)
            else:
                q_alpha1 = self.q_alpha1
            x = act_q1(x, q_alpha1)

        identity = x
        if self.tam is not None:
            x = self.tam(x)
            if bit != 32 or self.quantize_fp:
                act_q_tam = getattr(self, 'act_q%d' % bit)
                if self.switch_clipval:
                    q_alpha_tam = getattr(self, 'q_alpha_tam_%d' % bit)
                else:
                    q_alpha_tam = self.q_alpha_tam
                x = act_q_tam(x, q_alpha_tam)

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv1(x)
        else:
            out = self.conv1(x, self.max_bit_width, bit)

        if self.switch_bn:
            if bit != 32 or self.quantize_fp:
                bn1 = getattr(self, 'bn1_%d' % bit)
            else:
                bn1 = self.bn1
        else:
            bn1 = self.bn1
        out = bn1(out)
        out = self.relu(out)

        if bit != 32 or self.quantize_fp:
            act_q2 = getattr(self, 'act_q%d' % bit)
            if self.switch_clipval:
                q_alpha2 = getattr(self, 'q_alpha2_%d' % bit)
            else:
                q_alpha2 = self.q_alpha2
            out = act_q2(out, q_alpha2)

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv2(out)
        else:
            out = self.conv2(out, self.max_bit_width, bit)

        if self.switch_bn:
            if bit != 32 or self.quantize_fp:
                bn2 = getattr(self, 'bn2_%d' % bit)
            else:
                bn2 = self.bn2
        else:
            bn2 = self.bn2
        out = bn2(out)

        if self.downsample:
            if self.tam is not None:
                if 'TSM' in self.tam.name():
                    downsample = getattr(self, 'downsample_%d' % bit)
                    if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(identity)
                    else:
                        identity = downsample(identity, self.max_bit_width, bit)
                else:
                    downsample = getattr(self, 'downsample_%d' % bit)
                    if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(x)
                    else:
                        identity = downsample(x, self.max_bit_width, bit)
            else:
                downsample = getattr(self, 'downsample_%d' % bit)
                if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
                    identity = downsample(x)
                else:
                    identity = downsample(x, self.max_bit_width, bit)

        out += identity
        out = self.relu(out)

        return out, bit

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_frames,  bit_width_family, q_init=[10],
                 mean_aligned=False, switch_bn=False, switch_clipval=False, quantize_fp=True, stride=1, downsamples=None, temporal_module=None):
        super(Bottleneck, self).__init__()
        if len(q_init) == 1:
            q_init = q_init * len(bit_width_family)
        self.bit_width_family = bit_width_family
        self.max_bit_width = max(bit_width_family)
        self.switch_bn = switch_bn
        self.switch_clipval = switch_clipval
        self.quantize_fp = quantize_fp
        if self.switch_clipval:
            for a_idx, a_bit in enumerate(self.bit_width_family):
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
                    self.register_parameter('q_alpha1_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
                    self.register_parameter('q_alpha2_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
                    self.register_parameter('q_alpha3_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1), requires_grad=True))
        else:
            for a_bit in self.bit_width_family:
                if quantize_fp or a_bit != 32:
                    setattr(self, 'act_q%d' % a_bit, activation_quantize_fn2(a_bit, mean_aligned=mean_aligned))
            if not (len(bit_width_family) == 1 and bit_width_family[0] == 32 and not quantize_fp):
                self.register_parameter('q_alpha1', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha2', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha3', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not quantize_fp:
            self.conv1 = conv1x1_fp(inplanes, planes)
        else:
            self.conv1 = conv1x1(inplanes, planes, mean_aligned)
        if self.switch_bn:
            for w_bit in self.bit_width_family:
                if w_bit == 32 and not quantize_fp:
                    setattr(self, 'bn1', nn.BatchNorm2d(planes))
                else:
                    setattr(self, 'bn1_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn1 = nn.BatchNorm2d(planes)

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not quantize_fp:
            self.conv2 = conv3x3_fp(planes, planes, stride)
        else:
            self.conv2 = conv3x3(planes, planes, mean_aligned, stride)
        if self.switch_bn:
            for w_bit in self.bit_width_family:
                if w_bit == 32 and not quantize_fp:
                    self.bn2 = nn.BatchNorm2d(planes)
                else:
                    setattr(self, 'bn2_%d' % w_bit, nn.BatchNorm2d(planes))
        else:
            self.bn2 = nn.BatchNorm2d(planes)

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not quantize_fp:
            self.conv3 = conv1x1_fp(planes, planes * self.expansion)
        else:
            self.conv3 = conv1x1(planes, planes * self.expansion, mean_aligned)

        if self.switch_bn:
            for w_bit in self.bit_width_family:
                if w_bit == 32 and not quantize_fp:
                    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                else:
                    setattr(self, 'bn3_%d' % w_bit, nn.BatchNorm2d(planes * self.expansion))
        else:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()

        if downsamples is not None:
            assert len(downsamples) == len(self.bit_width_family)
            for w_idx, w_bit in enumerate(self.bit_width_family):
                setattr(self, 'downsample_%d' % w_bit, downsamples[w_idx])
            self.downsample = True
        else:
            self.downsample = False

        self.stride = stride

        if temporal_module is not None:
            self.tam = temporal_module(duration=num_frames, channels=inplanes)
            # TODO: to change
            if self.switch_clipval:
                for a_idx, a_bit in enumerate(self.bit_width_family):
                    if a_bit != 32 or quantize_fp:
                        self.register_parameter('q_alpha_tam_%d' % a_bit, nn.Parameter(q_init[a_idx] * torch.ones(1),
                                                                                   requires_grad=True))
            else:
                if not (len(bit_width_family) == 1 and bit_width_family[0] == 32 and not quantize_fp):
                    self.register_parameter('q_alpha_tam', nn.Parameter(q_init[0] * torch.ones(1), requires_grad=True))
        else:
            self.tam = None

    def forward(self, x, bit):
        if bit != 32 or self.quantize_fp:
            act_q1 = getattr(self, 'act_q%d' % bit)
            if self.switch_clipval:
                q_alpha1 = getattr(self, 'q_alpha1_%d' % bit)
            else:
                q_alpha1 = self.q_alpha1
            x = act_q1(x, q_alpha1)

        identity = x
        if self.tam is not None:
            x = self.tam(x)
            if bit != 32 or self.quantize_fp:
                act_q_tam = getattr(self, 'act_q%d' % bit)
                if self.switch_clipval:
                    q_alpha_tam = getattr(self, 'q_alpha_tam_%d' % bit)
                else:
                    q_alpha_tam = self.q_alpha_tam
                x = act_q_tam(x, q_alpha_tam)

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv1(x)
        else:
            out = self.conv1(x, self.max_bit_width, bit)

        if self.switch_bn:
            if bit != 32 or self.quantize_fp:
                bn1 = getattr(self, 'bn1_%d' % bit)
            else:
                bn1 = self.bn1
        else:
            bn1 = self.bn1
        out = bn1(out)
        out = self.relu(out)

        if bit != 32 or self.quantize_fp:
            act_q2 = getattr(self, 'act_q%d' % bit)
            if self.switch_clipval:
                q_alpha2 = getattr(self, 'q_alpha2_%d' % bit)
            else:
                q_alpha2 = self.q_alpha2
            out = act_q2(out, q_alpha2)

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv2(out)
        else:
            out = self.conv2(out, self.max_bit_width, bit)

        if self.switch_bn:
            if bit != 32 or self.quantize_fp:
                bn2 = getattr(self, 'bn2_%d' % bit)
            else:
                bn2 = self.bn2
        else:
            bn2 = self.bn2
        out = bn2(out)
        out = self.relu(out)

        if bit != 32 or self.quantize_fp:
            act_q3 = getattr(self, 'act_q%d' % bit)
            if self.switch_clipval:
                q_alpha3 = getattr(self, 'q_alpha3_%d' % bit)
            else:
                q_alpha3 = self.q_alpha3
            out = act_q3(out, q_alpha3)

        if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
            out = self.conv3(out)
        else:
            out = self.conv3(out, self.max_bit_width, bit)

        if self.switch_bn:
            if bit != 32 or self.quantize_fp:
                bn3 = getattr(self, 'bn3_%d' % bit)
            else:
                bn3 = self.bn3
        else:
            bn3 = self.bn3
        out = bn3(out)

        if self.downsample:
            if self.tam is not None:
                if 'TSM' in self.tam.name():
                    downsample = getattr(self, 'downsample_%d' % bit)
                    if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(identity)
                    else:
                        identity = downsample(identity, self.max_bit_width, bit)
                else:
                    downsample = getattr(self, 'downsample_%d' % bit)
                    if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
                        identity = downsample(x)
                    else:
                        identity = downsample(x, self.max_bit_width, bit)
            else:
                downsample = getattr(self, 'downsample_%d' % bit)
                if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not self.quantize_fp:
                    identity = downsample(x)
                else:
                    identity = downsample(x, self.max_bit_width, bit)

        out += identity
        out = self.relu(out)

        return out, bit


class ResNet_PACT_Backbone(nn.Module):
    def __init__(self, depth, num_frames, bit_width_family, q_init, mean_aligned=False,
                 switch_bn=False, switch_clipval=False, is_32fp=False,
                 num_classes=1000, zero_init_residual=False,
                 without_t_stride=False, temporal_module=None, fpn_dim=-1, pooling_method='max',
                 input_channels=3, tam_pos='all'):
        super(ResNet_PACT_Backbone, self).__init__()
        self.bit_width_family = bit_width_family
        self.mean_aligned = mean_aligned
        self.switch_bn = switch_bn
        self.is_32fp = is_32fp
        self.switch_clipval = switch_clipval
        self.bit_list = "".join([str(bit) for bit in self.bit_width_family])
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

        self.depth = depth
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

        if self.is_32fp:
            self.inplanes = 64
            self.switch_bn, self.switch_clipval = False, False
            self.bit_width_family = [32]
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
            self.bit_width_family = bit_width_family

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

        for name, param in self.named_parameters():
            if '32fp' in name:
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, skip_tam=False, quantize_fp=True):
        downsamples = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsamples = []
            if len(self.bit_width_family) == 1 and self.bit_width_family[0] == 32 and not quantize_fp:
                conv_layer = conv1x1_fp(self.inplanes, planes * block.expansion, stride)
            else:
                conv_layer = conv1x1(self.inplanes, planes * block.expansion, self.mean_aligned, stride)
            if self.switch_bn:
                for _ in self.bit_width_family:
                    downsamples.append(mySequential(conv_layer,
                    nn.BatchNorm2d(planes * block.expansion),
                    ))
            else:
                bn = nn.BatchNorm2d(planes * block.expansion)
                for _ in self.bit_width_family:
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
        block_tmp = block(self.inplanes, planes, self.num_frames, self.bit_width_family,
                          self.q_init, self.mean_aligned, self.switch_bn, self.switch_clipval, quantize_fp, stride,
                          downsamples, temporal_module=self.temporal_module if not skip_tam_1 else None)
        layers.append(block_tmp)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            block_tmp = block(self.inplanes, planes, self.num_frames, self.bit_width_family,
                              self.q_init,  self.mean_aligned, self.switch_bn, self.switch_clipval, quantize_fp,
                              temporal_module=self.temporal_module if not skip_tam_2 else None)
            layers.append(block_tmp)

        return mySequential(*layers)

    def extract_feature(self, x, bit,  last_layer=True):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        fp1 = self.maxpool(x)

        fp2, _ = self.layer1(fp1, bit)
        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3, _ = self.layer2(fp2_d, bit)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4, _ = self.layer3(fp3_d, bit)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5, _ = self.layer4(fp4_d, bit)

        if last_layer is True:
            return fp5
        else:
            return fp1, fp2_d, fp3_d, fp4_d, fp5

    def forward_32fp(self, x, batch_size, bit):
        x = self.conv1_32fp(x)
        x = self.bn1_32fp(x)
        x = self.relu(x)
        fp1 = self.maxpool(x)

        fp2, _ = self.layer1_32fp(fp1, bit)
        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3, _ = self.layer2_32fp(fp2_d, bit)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4, _ = self.layer3_32fp(fp3_d, bit)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5, _ = self.layer4_32fp(fp4_d, bit)

        x_32fp = self.avgpool(fp5)
        x_32fp = x_32fp.view(x_32fp.size(0), -1)
        x_32fp = self.fc_32fp(x_32fp)

        n_t, c = x_32fp.shape
        out_32fp = x_32fp.view(batch_size, -1, c)

        # average the prediction from all frames
        out_32fp = torch.mean(out_32fp, dim=1)
        return out_32fp, fp5

    def forward(self, x, bit):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)

        if self.is_32fp and max(self.bit_width_family) == bit:
            with torch.no_grad():
                x_32fp = x.clone()
                out_32fp, fp5_32fp = self.forward_32fp(x_32fp, batch_size, 32)
        else:
            out_32fp, fp5_32fp = None, None

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        fp1 = self.maxpool(x)
        fp2, _ = self.layer1(fp1, bit)
        fp2_d = self.pool1(fp2) if not self.without_t_stride else fp2
        fp3, _ = self.layer2(fp2_d, bit)
        fp3_d = self.pool2(fp3) if not self.without_t_stride else fp3
        fp4, _ = self.layer3(fp3_d, bit)
        fp4_d = self.pool3(fp4) if not self.without_t_stride else fp4
        fp5, _ = self.layer4(fp4_d, bit)

        return fp5, out_32fp, fp5_32fp

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]


class ResNetVideoIQ(nn.Module):
    def __init__(self, depth, num_frames, bit_width_family,  p_bit, q_init, skip_list, mean_aligned=False,
                 switch_bn=False, switch_clipval=False, is_32fp=False, is_policy_pred=False, use_fp_as_bb=False,
                 num_classes=1000, dropout=0.5, zero_init_residual=False,
                 without_t_stride=False, temporal_module=None, fpn_dim=-1, pooling_method='max',
                 input_channels=3, tam_pos='all'):
        super(ResNetVideoIQ, self).__init__()
        block = BasicBlock if depth < 50 else Bottleneck
        self.prec_dim = len(bit_width_family) + 1 if use_fp_as_bb else len(bit_width_family)
        self.skip_dim = len(skip_list)
        self.is_policy_pred = is_policy_pred
        self.is_32fp = is_32fp or use_fp_as_bb
        self.use_fp_as_bb = use_fp_as_bb
        self.temporal_module = temporal_module
        self.bit_width_family = bit_width_family
        self.bit_list = "".join([str(bit) for bit in self.bit_width_family])
        self.skip_list = "".join([str(f) for f in skip_list])
        self.p_bit = p_bit
        self.depth = depth
        self.is_policy_pred = is_policy_pred
        self.use_fp_as_bb = use_fp_as_bb
        self.without_t_stride = without_t_stride
        self.fpn_dim = fpn_dim
        if use_fp_as_bb:
            self.bit_list = "32" + self.bit_list
        self.q_init = q_init
        uniques = np.unique(self.q_init)
        if len(uniques) == 1:
            self.q_init_list = str(self.q_init[0])
        else:
            self.q_init_list = "".join([str(bit) for bit in self.q_init])
        self.backbone = ResNet_PACT_Backbone(depth, num_frames, bit_width_family, q_init,
                                             mean_aligned, switch_bn, switch_clipval, self.is_32fp, num_classes, zero_init_residual,
                                             without_t_stride, temporal_module, fpn_dim, pooling_method, input_channels,
                                             tam_pos)
        self.policy_net = MobileNetV2(num_frames, self.prec_dim, 512, 0.8, skip_list=skip_list, bit=p_bit,
                                      q_init=10, is_policy_pred=is_policy_pred, num_classes=num_classes, width_mult=1.)
        self.num_frames = num_frames
        self.p_bit = p_bit
        self.bit_width_family = bit_width_family
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_policynet_backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'policy_net' in name:
                if 'q_alpha' not in name:
                    if 'rnn' not in name and 'linear' not in name and 'fc' not in name:
                        params.append(param)

        return params

    def get_policynet_leaf_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'policy_net' in name:
                if 'q_alpha' not in name:
                    if 'rnn' in name or 'linear' in name or 'fc' in name:
                        params.append(param)

        return params

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'policy_net' not in name:
                param.requires_grad = False

    def freeze_quant_params(self):
        for name, param in self.named_parameters():
            if 'q_alpha' in name:
                param.requires_grad = False

    def free_quant_params(self):
        for name, param in self.named_parameters():
            if 'q_alpha' in name:
                param.requires_grad = True

    def free_backbone(self):
        for name, param in self.named_parameters():
            if 'backbone' in name and '32fp' not in name:
                param.requires_grad = True
            if 'fc' in name and 'policy_net' not in name:
                param.requires_grad = True

    def freeze_policynet(self):
        for name, param in self.named_parameters():
            if 'policy_net' in name:
                param.requires_grad = False

    def free_policynet(self):
        for name, param in self.named_parameters():
            if 'policy_net' in name:
                param.requires_grad = True

    def combine_logits(self, r, base_out_list):
        # TODO r                N, T, K
        # TODO base_out_list  < K * (N, T, C)
        pred_tensor = torch.stack(base_out_list, dim=2)
        # print('pred_tensor.shape', pred_tensor.shape)
        r_tensor = r[:, :, :self.prec_dim].unsqueeze(-1).expand_as(pred_tensor)
        t_tensor = torch.sum(r[:, :, :self.prec_dim], dim=[1, 2]).clamp(1) / float(self.num_frames) # TODO sum T, K to count frame
        feat = (pred_tensor * r_tensor).sum(dim=[2])
        t_tensor = t_tensor.unsqueeze(-1).unsqueeze(-1).expand_as(feat)
        return feat / t_tensor

    def set_temperature(self, temperature):
        self.policy_net.set_temperature(temperature)

    def decay_temperature(self, decay_ratio=None):
        self.policy_net.decay_temperature(decay_ratio)

    @staticmethod
    def r_policy(batch_size, num_frames, decision_dim, device_id, dist=None):
        device_id = device_id if device_id >= 0 else 'cpu'
        if dist is None:
            p = torch.randint(0, decision_dim, size=[batch_size, num_frames], device=device_id)
        else:
            assert len(dist) == decision_dim
            tmp = torch.rand(size=[batch_size, num_frames], device=device_id)
            p = torch.zeros(size=[batch_size, num_frames], device=device_id)
            dist = np.cumsum(dist).tolist()
            for d in dist[:-1]:
                desired_tensor = torch.where(tmp < d, torch.tensor(0.).to(device_id), torch.tensor(1.).to(device_id))
                p += desired_tensor
        labels = list(range(decision_dim))
        p = one_hot(p, labels)
        return p

    def ada_forward(self, x, rand_policy=False, dist=None, deterministic=False):
        batch_size, c_t, h, w = x.shape
        policy_out = None
        if rand_policy:
            device_id = x.get_device()
            decision_dim = self.prec_dim + self.skip_dim
            r = self.r_policy(batch_size, self.num_frames, decision_dim, device_id, dist=dist)
            p = r
        else:
            x_policy = F.interpolate(x, size=[84, 84], mode='bilinear')
            r, p, policy_out = self.policy_net(x_policy, deterministic=deterministic)

        base_feat_list = []
        output_32fp = None
        for idx, bit in enumerate(self.bit_width_family):
            if idx == 0:
                feat, output_32fp, feat_32fp = self.backbone(x, bit)
                # print('feat size: ', feat.shape)
                # print('feat 32fp size: ', feat_32fp.shape)
                if self.use_fp_as_bb:
                    feat_32fp = self.avgpool(feat_32fp)
                    feat_32fp = feat_32fp.view(feat.size(0), -1)
                    base_feat_list.append(feat_32fp.view(batch_size, self.num_frames, -1))
            else:
                feat, _, _ = self.backbone(x, bit)
            feat = self.avgpool(feat)
            feat = feat.view(feat.size(0), -1)
            base_feat_list.append(feat.view(batch_size, self.num_frames, -1))

        output = self.combine_logits(r, base_feat_list).view(batch_size * self.num_frames, -1)
        x = self.dropout(output)
        x = self.fc(x)
        n_t, c = x.shape
        out = x.view(batch_size, -1, c)
        out = torch.mean(out, dim=1)
        with torch.no_grad():
            x_h = self.dropout(base_feat_list[0].view(batch_size * self.num_frames, -1))
            x_h = self.fc(x_h)
            out_h = x_h.view(batch_size, -1, c)
            out_h = torch.mean(out_h, dim=1)
        return out, r, p, out_h, output_32fp, policy_out


    def ap_forward(self, x, bit=4):
        batch_size, c_t, h, w = x.shape

        output_32fp = None
        if max(self.bit_width_family) == bit:
            feat, output_32fp, _ = self.backbone(x, bit)
        else:
            feat, _, _ = self.backbone(x, bit)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        x = self.dropout(feat)
        x = self.fc(x)
        n_t, c = x.shape
        out = x.view(batch_size, -1, c)
        out = torch.mean(out, dim=1)

        return out, output_32fp

    def forward(self, x, is_ada=True, rand_policy=False, dist=None, deterministic=False, w_bit=4, a_bit=4):
        if is_ada:
            return self.ada_forward(x, rand_policy=rand_policy, dist=dist, deterministic=deterministic)
        else:
            return self.ap_forward(x, w_bit)

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
        name += 'resnet_videoiq-{}-b{}s{}init{}-p{}'.format(self.depth, self.bit_list, self.skip_list, self.q_init_list, self.p_bit)
        if self.is_policy_pred:
            name += '-policy_pred'
        if self.use_fp_as_bb:
            name += '-use_fp_as_bb'
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)
        if self.fpn_dim > 0:
            name += "-fpn{}".format(self.fpn_dim)

        return name


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


def resnet_videoiq(depth, num_classes, without_t_stride, groups, temporal_module_name,
                          dw_conv, blending_frames, blending_method, dropout, fpn_dim, pooling_method,
                          input_channels, tam_pos, p_bit, q_init, bit_width_family, skip_list, mean_aligned,
                          switch_bn, switch_clipval, is_32fp, is_policy_pred, use_fp_as_bb,  imagenet_pretrained=True, resnet_imagenet_path=None,
                          mobilenet_imagenet_path=None,  **kwargs):

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

    model = ResNetVideoIQ(depth, num_frames=groups, bit_width_family=bit_width_family, skip_list=skip_list, p_bit=p_bit,
                                q_init=q_init, mean_aligned=mean_aligned, switch_bn=switch_bn, switch_clipval=switch_clipval,
                                 is_32fp=is_32fp, is_policy_pred=is_policy_pred, use_fp_as_bb=use_fp_as_bb,
                                num_classes=num_classes, without_t_stride=without_t_stride, temporal_module=temporal_module,
                                dropout=dropout, fpn_dim=fpn_dim, pooling_method=pooling_method,
                                input_channels=input_channels, tam_pos=tam_pos)

    if add_non_local:
        model = make_non_local_v2(model, groups)

    if imagenet_pretrained:
        # load the resnet pretrain
        if resnet_imagenet_path is None:
            state_dict = model_zoo.load_url(model_urls['resnet{}'.format(depth)], map_location='cpu')
            old_state_dict = model.state_dict()
            for k, v in old_state_dict.items():
                if k in state_dict.keys():
                    old_state_dict[k] = state_dict[k]
                else:
                    tokens = k.split('.')
                    new_tokens = []
                    for t in tokens:
                        if 'downsample' in t or 'bn' in t:
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
        else:
            print('Loading snapshots from ', resnet_imagenet_path)
            state_dict = torch.load(resnet_imagenet_path, map_location='cpu')['state_dict']
            new_state_dict = {}
            # model_param_names = model.state_dict().keys()
            for k, v in state_dict.items():
                # new_k = k.replace("module.", "")
                # if new_k in model_param_names and "num_batches_tracked" not in new_k:
                new_state_dict[k.replace("module.", "")] = v

            old_state_dict = model.backbone.state_dict()
            for k, v in old_state_dict.items():
                if k in new_state_dict.keys():
                    old_state_dict[k] = new_state_dict[k]
                else:
                    tokens = k.split('.')
                    new_tokens = []
                    for t in tokens:
                        if 'downsample' in t or 'bn' in t:
                            subtokens = t.split('_')
                            if len(subtokens) == 2:
                                new_tokens.append(subtokens[0])
                            else:
                                new_tokens.append(t)
                        else:
                            new_tokens.append(t)
                    new_k = '.'.join(new_tokens)
                    if new_k in new_state_dict.keys():
                        old_state_dict[k] = new_state_dict[new_k]
                    else:
                        print('Skip %s, new k is %s' % (k, new_k))
            state_dict = old_state_dict
            fc_state_dict = model.fc.state_dict()
            for k, v in fc_state_dict.items():
                new_k = 'fc.' + k
                if new_k in new_state_dict.keys():
                    fc_state_dict[k] = new_state_dict[new_k]
                else:
                    print('Skip %s' % (k))
            model.fc.load_state_dict(fc_state_dict, strict=False)
        if input_channels != 3:  # convert the RGB model to others, like flow
            state_dict = convert_rgb_model_to_others(state_dict, input_channels, 7)
        model.backbone.load_state_dict(state_dict, strict=False)

        # load the mobilenet
        if mobilenet_imagenet_path is not None:
            print('Loading policy network snapshots from ', mobilenet_imagenet_path)
            state_dict = torch.load(mobilenet_imagenet_path, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                new_state_dict = {}
                state_dict = state_dict['state_dict']
                for k, v in state_dict.items():
                    # new_k = k.replace("module.", "")
                    # if new_k in model_param_names and "num_batches_tracked" not in new_k:
                    new_state_dict[k.replace("module.", "")] = v
                state_dict = new_state_dict

            missing_keys, unexpected_keys = model.policy_net.load_state_dict(state_dict, strict=False)

            print('Missing Keys: ', missing_keys)
            print('Unexpected Keys: ', unexpected_keys)

    return model


if __name__ == '__main__':
    from torchsummary import torchsummary
    model = resnet_videoiq(18, 187, without_t_stride=True, groups=8, dropout=0.5,
                                  temporal_module_name='TSN', dw_conv=True, blending_frames=3, tam_pos='half_2',
                                  blending_method='sum', fpn_dim=-1, pooling_method='max', input_channels=3, bit_width_family=[2, 4],
                                  skip_list=[1, 2, 4], q_init=[2], mean_aligned=True, switch_bn=True, switch_clipval=True,
                                  imagenet_pretrained=False)

    # dummy_data = (24, 224, 224)
    model.eval()
    # model_summary = torchsummary.summary(model, input_size=dummy_data)
    print(model)
    # print(model_summary)
    # print(model.network_name)
    input_dummy = torch.ones(1, 24, 224, 224)
    output = model(input_dummy)

    # flops for the entire backbone
    from utils.utils import compute_flops
    dummy_input = torch.ones(1, 24, 224, 224)
    flops = compute_flops(model, dummy_input, {})
    print(flops)

    # compute first layer flops in the backbone
    from utils.utils import compute_flops
    dummy_input = torch.ones(1, 24, 84, 84)
    flops = compute_flops(model, dummy_input, {})
    print(flops)

    # compute last layer flops in the backbone
