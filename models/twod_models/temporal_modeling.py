
# author: https://github.com/Pika7ma/Temporal-Shift-Module/blob/master/tsm_util.py

import torch
import torch.nn.functional as F
import torch.nn as nn


class SEModule(nn.Module):

    def __init__(self, channels, dw_conv):
        super().__init__()
        ks = 1
        pad = (ks - 1) // 2
        self.fc1 = nn.Conv2d(channels, channels, kernel_size=ks,
                             padding=pad, groups=channels if dw_conv else 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x

class TSMModule(nn.Module):

    def __init__(self, duration, blending_frames=3):
        super().__init__()
        self.blending_frames = blending_frames
        self.duration = duration
        self.fold_div = 8
        self.tsm_padding = 'zero'

    def forward(self, x):
        return tsm(x, self.duration, self.tsm_padding, self.fold_div, self.blending_frames)

    def name(self):
        return "TSM-b{}".format(self.blending_frames)


class TAMslow(nn.Module):

    def __init__(self, duration, channels, blending_frames=3, blending_method='sum'):
        super().__init__()
        self.blending_frames = blending_frames
        self.blending_method = blending_method

        self.tam = nn.Conv2d(channels, channels * blending_frames, kernel_size=1,
                             padding=0, bias=False, groups=channels)
        self.relu = nn.ReLU(inplace=True)
        self.duration = duration

    def forward(self, x):
        nt, c, h, w = x.shape
        new_x = self.tam(x)
        prev_x = new_x[:, 0:c * self.blending_frames: self.blending_frames, ...]
        curr_x = new_x[:, 1:c * self.blending_frames: self.blending_frames, ...]
        next_x = new_x[:, 2:c * self.blending_frames: self.blending_frames, ...]

        prev_x = prev_x.view((-1, self.duration) + prev_x.size()[1:])
        curr_x = curr_x.view((-1, self.duration) + curr_x.size()[1:])
        next_x = next_x.view((-1, self.duration) + next_x.size()[1:])

        prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
        next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

        out = torch.stack([prev_x, curr_x, next_x], dim=0)
        out = torch.sum(out, dim=0)
        out = self.relu(out)
        # [N, T, C, N, H]
        # n, t, c, h, w = out.shape
        out = out.view((-1, ) + out.size()[2:])

        return out

class TAM3D(nn.Module):

    def __init__(self, duration, channels, blending_frames=3, blending_method='sum'):
        super().__init__()
        self.blending_frames = blending_frames
        self.blending_method = blending_method

        self.tam = nn.Conv3d(channels, channels, kernel_size=(blending_frames, 1, 1),
                             padding=(1, 0, 0), bias=False, groups=channels)
        self.relu = nn.ReLU(inplace=True)
        self.duration = duration
        nn.init.kaiming_normal_(self.tam.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        nt, c, h, w = x.shape
        x = x.view((-1, self.duration) + x.size()[1:]).transpose(1, 2)
        x = self.tam(x)
        x = self.relu(x)
        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
        return x

class TAM(nn.Module):

    def __init__(self, duration, channels, dw_conv=True, blending_frames=3, blending_method='sum'):
        super().__init__()
        self.blending_frames = blending_frames
        self.blending_method = blending_method

        if blending_frames == 3:
            self.prev_se = SEModule(channels, dw_conv)
            self.next_se = SEModule(channels, dw_conv)
            self.curr_se = SEModule(channels, dw_conv)
        else:
            self.blending_layers = nn.ModuleList([SEModule(channels, dw_conv) for _ in range(blending_frames)])
        self.relu = nn.ReLU(inplace=True)
        self.duration = duration

    def name(self):
        return "TAM-b{}-{}".format(self.blending_frames, self.blending_method)

    def forward(self, x):
        if self.blending_method == 'maxnorm':
            prev_x = self.prev_se(x)
            curr_x = self.curr_se(x)
            next_x = self.next_se(x)

            '''max pooling'''
            prev_x = F.max_pool2d(prev_x, kernel_size=prev_x.size()[-2:])
            curr_x = F.max_pool2d(curr_x, kernel_size=curr_x.size()[-2:])
            next_x = F.max_pool2d(next_x, kernel_size=next_x.size()[-2:])

            # shift channels
            prev_x = prev_x.view((-1, self.duration) + prev_x.size()[1:])
            curr_x = curr_x.view((-1, self.duration) + curr_x.size()[1:])
            next_x = next_x.view((-1, self.duration) + next_x.size()[1:])

            prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
            next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

            # remove the last dimension
            out = torch.stack([prev_x, curr_x, next_x], dim=0).squeeze(5)
            # remove negative activations
            out = self.relu(out)
            # expand dimensions to [..., W, H]
            out = out.expand(out.size()[:-1] + (x.size()[-1] * x.size()[-2],))
            out = out.view(out.size()[:-1] + x.size()[-2:])
            # normalize activations via softmax
            out = F.softmax(out, dim = 0)

            # shift feature channels
            x_tmp = x.view((-1, self.duration) + x.size()[1:])
            prev_x = F.pad(x_tmp, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
            next_x = F.pad(x_tmp, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

            # weighted sum
            out = out * torch.stack([prev_x, x_tmp, next_x], dim=0)
            out = torch.sum(out, dim = 0)

            out = out.view((-1,) + out.size()[2:])

            return out

        else:
            if self.blending_frames == 3:
                prev_x = self.prev_se(x)
                curr_x = self.curr_se(x)
                next_x = self.next_se(x)
                prev_x = prev_x.view((-1, self.duration) + prev_x.size()[1:])
                curr_x = curr_x.view((-1, self.duration) + curr_x.size()[1:])
                next_x = next_x.view((-1, self.duration) + next_x.size()[1:])

                prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
                next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

                out = torch.stack([prev_x, curr_x, next_x], dim=0)
            else:
                # multiple blending
                xs = [se(x) for se in self.blending_layers]
                xs = [x.view((-1, self.duration) + x.size()[1:]) for x in xs]

                shifted_xs = []
                for i in range(self.blending_frames):
                    shift = i - (self.blending_frames // 2)
                    x_temp = xs[i]
                    n, t, c, h, w = x_temp.shape
                    start_index = 0 if shift < 0 else shift
                    end_index = t if shift < 0 else t + shift
                    padding = None
                    if shift < 0:
                        padding = (0, 0, 0, 0, 0, 0, abs(shift), 0)
                    elif shift > 0:
                        padding = (0, 0, 0, 0, 0, 0, 0, shift)
                    shifted_xs.append(F.pad(x_temp, padding)[:, start_index:end_index, ...]
                                      if padding is not None else x_temp)

                out = torch.stack(shifted_xs, dim=0)

        if self.blending_method == 'sum':
            out = torch.sum(out, dim=0)
        elif self.blending_method == 'max':
            out, _ = torch.max(out, dim=0)
        elif self.blending_method == 'sumnorm':
            out = F.softmax(out, dim = 0)
            x_tmp = x.view((-1, self.duration) + x.size()[1:])
            prev_x = F.pad(x_tmp, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
            next_x = F.pad(x_tmp, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

            out = out * torch.stack([prev_x, x_tmp, next_x], dim=0)
            out = torch.sum(out, dim = 0)

        else:
            raise ValueError('Blending method %s not supported' % (self.blending_method))

        out = self.relu(out)
        # [N, T, C, N, H]
        n, t, c, h, w = out.shape
        out = out.view((-1, ) + out.size()[2:])
        # out = out.contiguous()

        return out

class GroupTAM(nn.Module):

    def __init__(self, groups, channels, expand=False, dw_conv=True):
        super().__init__()
        self.groups = groups
        self.channels = channels
        self.prev_se = SEModule(channels // groups, dw_conv)
        self.next_se = SEModule(channels // groups, dw_conv)
        self.curr_se = SEModule(channels // groups, dw_conv)
        self.relu = nn.ReLU(inplace=True)
        self.expand = expand
        if self.expand:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.channels // groups * 3, self.channels // groups,  kernel_size=1, stride=1, groups = groups, bias = False),
                nn.BatchNorm2d(self.channels // groups),)


    def name(self):
        return "GroupTAM".format() if not self.expand else "GroupExpandTAM"

    def forward(self, x):

        n, c, w, h = x.shape
        x = x.view (n * self.groups, c//self.groups, w, h)

        prev_x = self.prev_se(x)
        curr_x = self.curr_se(x)
        next_x = self.next_se(x)

        prev_x = prev_x.view((-1, self.groups) + prev_x.size()[1:])
        curr_x = curr_x.view((-1, self.groups) + curr_x.size()[1:])
        next_x = next_x.view((-1, self.groups) + next_x.size()[1:])

        prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
        next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

        if not self.expand:
            out = torch.stack([prev_x, curr_x, next_x], dim=0)
            out = torch.sum(out, dim=0)
            out = self.relu(out)
        else:
            # Nx3*Fx
            #print (prev_x.shape)
            #out = torch.stack([prev_x, curr_x, next_x], dim=0)
      #      print (prev_x.shape)
            out = torch.cat((prev_x, curr_x, next_x), dim=2)
      #      print (out.shape)
            out = out.view(n*self.groups, -1, w, h)
      #      print (out.shape)
            out = self.downsample(out)
      #      print (out.shape)
            out = self.relu(out)

        #print (out.shape)
        out = out.view(n, c, w, h)

        return out

'''
class GroupTAM(nn.Module):

    def __init__(self, groups, channels, dw_conv=True):
        super().__init__()
        self.groups = groups
        self.channels = channels
        self.prev_se = SEModule(channels // groups, dw_conv)
        self.next_se = SEModule(channels // groups, dw_conv)
        self.curr_se = SEModule(channels // groups, dw_conv)
        self.relu = nn.ReLU(inplace=True)

    def name(self):
        return "GroupTAM".format()

    def forward(self, x):

        n, c, w, h = x.shape
        x = x.view (n * self.groups, c//self.groups, w, h)

        prev_x = self.prev_se(x)
        curr_x = self.curr_se(x)
        next_x = self.next_se(x)

        prev_x = prev_x.view((-1, self.groups) + prev_x.size()[1:])
        curr_x = curr_x.view((-1, self.groups) + curr_x.size()[1:])
        next_x = next_x.view((-1, self.groups) + next_x.size()[1:])

        prev_x = F.pad(prev_x, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
        next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]

        out = torch.stack([prev_x, curr_x, next_x], dim=0)
        out = torch.sum(out, dim=0)
        out = self.relu(out)

        #print (out.shape)
        out = out.view(n, c, w, h)

        return out
'''

class LearnableTemporalFusion(nn.Module):

    def __init__(self, duration, channels, dw_conv=True, blending_frames=3):
        super().__init__()
        self.blending_frames = blending_frames

        if blending_frames == 3:
            self.prev_se = SEModule(channels, dw_conv)
            self.next_se = SEModule(channels, dw_conv)
            self.curr_se = SEModule(channels, dw_conv)
        self.relu = nn.ReLU(inplace=True)
        self.duration = duration

    def forward(self, x):
        n = x.size()[0]
        prev_x = self.prev_se(x[range(0, n ,2), ::])
        curr_x = self.curr_se(x[range(1, n, 2), ::])
        next_x = self.prev_se(x[range(0, n, 2), ::])

        prev_x = prev_x.view((-1, self.duration) + prev_x.size()[1:])
        curr_x = curr_x.view((-1, self.duration) + curr_x.size()[1:])
        next_x = next_x.view((-1, self.duration) + next_x.size()[1:])

        next_x = F.pad(next_x, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]
        next_x[:,-1,::] = next_x[:,-2,::]

        out = torch.stack([prev_x, curr_x, next_x], dim=0)
        out = torch.sum(out, dim = 0)
        out = self.relu(out)
        out = out.view((-1,) + out.size()[2:])

        return out

def tsm(tensor, duration, version='zero', fold_div=8, blending_frames=3):
    # tensor [N*T, C, H, W]

    # official implementation seems to be slower
    nt, c, h, w = tensor.size()
    n_batch = nt // duration
    #print (nt, c, h, w, duration, n_batch)
    x = tensor.view(n_batch, duration, c, h, w)

    out = torch.zeros_like(x)
    fold = c // fold_div
    if blending_frames == 3:
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
    elif blending_frames == 5:
        out[:, :-2, :fold] = x[:, 2:, :fold]  # 2-shift left
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]  # 1-shift left
        out[:, 2:, 2*fold:3*fold] = x[:, :-2, 2*fold:3*fold]  # 1-shift right
        out[:, 1:, 3*fold:4*fold] = x[:, :-1, 3*fold:4*fold]  # 2-shift right
        out[:, :, 4*fold:] = x[:, :, 4*fold:]  # not shift
    elif blending_frames == 7:
        out[:, :-3, :fold] = x[:, 3:, :fold]  # 2-shift left
        out[:, :-2, fold:2*fold] = x[:, 2:, fold:2*fold]  # 2-shift left
        out[:, :-1, 2*fold:3*fold] = x[:, 1:, 2*fold:3*fold]  # 1-shift left
        out[:, 3:, 3*fold:4*fold] = x[:, :-3, 3*fold:4*fold]  # 1-shift right
        out[:, 2:, 4*fold:5*fold] = x[:, :-2, 4*fold:5*fold]  # 2-shift right
        out[:, 1:, 5*fold:6*fold] = x[:, :-1, 5*fold:6*fold]  # 2-shift right
        out[:, :, 6*fold:] = x[:, :, 6*fold:]  # not shift
    else:
        raise ValueError('Blending than 7 frames is not supported yet!')

    return out.view(nt, c, h, w)

    # size = tensor.size()
    # tensor = tensor.view((-1, duration) + size[1:])
    # # tensor [N, T, C, H, W]
    # pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 8,
    #                                                      size[1] // 8,
    #                                                      size[1] // 8 * 6], dim=2)
    # if version == 'zero':
    #     pre_tensor = F.pad(pre_tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
    #     post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]
    # elif version == 'circulant':
    #     pre_tensor = torch.cat((pre_tensor[:, -1:, ...],
    #                             pre_tensor[:, :-1, ...]), dim=1)
    #     post_tensor = torch.cat((post_tensor[:, 1:, ...],
    #                              post_tensor[:, :1, ...]), dim=1)
    # else:
    #     raise ValueError('Unknown TSM version: {}'.format(version))
    # return torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size)


class SegmentConsensusIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor):
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SegmentConsensusAvg(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor):
        shape = input_tensor.size()
        ctx.save_for_backward(shape)
        output = input_tensor.mean(dim=1, keepdim=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape, = ctx.saved_tensors
        grad_in = grad_output.expand(shape) / float(shape[1])
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        if self.consensus_type == 'avg':
            self.consensus_module = SegmentConsensusAvg()
        elif self.consensus_type == 'identity':
            self.consensus_module = SegmentConsensusIdentity()

    def forward(self, input):
        return self.consensus_module(input)


def temporal_modeling_module(name, duration, channels, dw_conv=True,
                             blending_frames=3, blending_method='sum'):
    if name is None:
        return None

    if name == 'TSM':  # the original one
        return TSMModule(duration, blending_frames)
    elif name == 'TAM':
        return TAM(duration, channels, dw_conv, blending_frames, blending_method)
    elif name == 'GroupTAM':
        return GroupTAM(duration, channels, expand=False, dw_conv=dw_conv)
    elif name == 'GroupExpTAM':
        return GroupTAM(duration, channels, expand=True, dw_conv=dw_conv)
    elif name == 'LearnableTemporalFusion':
        return LearnableTemporalFusion(duration, channels, dw_conv, blending_frames)
    elif name == 'NonLocal':
        from .ops.non_local import NL3DWrapper
        return NL3DWrapper(channels, duration)
    else:
        raise ValueError('incorrect tsm module name %s' % name)

