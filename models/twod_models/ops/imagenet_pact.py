import torch
import torch.nn as nn
import torch.nn.functional as F


def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val)
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    with torch.no_grad():
        scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
        scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
        is_scalar = scalar_min and scalar_max

        if scalar_max and not scalar_min:
            sat_max = sat_max.to(sat_min.device)
        elif scalar_min and not scalar_max:
            sat_min = sat_min.to(sat_max.device)

        #        print('device {}, sat_min {}'.format(sat_min.device.index, sat_min))
        #        print('device {}, sat_max {}'.format(sat_min.device.index, sat_max))

        # if any(sat_min > sat_max):
        #     raise ValueError('saturation_min must be smaller than saturation_max, sat_min={}, sat_max={}'.format(sat_min, sat_max))

        n = 2 ** num_bits - 1

        # Make sure 0 is in the range
        sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
        sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

        diff = sat_max - sat_min
        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        diff[diff == 0] = n

        scale = n / diff
        # pdb.set_trace()
        zero_point = scale * sat_min
        if integral_zero_point:
            zero_point = zero_point.round()
        if signed:
            zero_point += 2 ** (num_bits - 1)
        if is_scalar:
            return scale.item(), zero_point.item()
        # pdb.set_trace()
        #        print('device {}, scale {}'.format(scale.device.index, scale))
        #        print('device {}, zero_point {}'.format(zero_point.device.index, zero_point))
        return scale, zero_point


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    # pdb.set_trace()
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    # pdb.set_trace()
    if isinstance(scale, torch.Tensor):
        # pdb.set_trace()
        return torch.round(scale.to(input.device) * input - zero_point.to(input.device))  # HACK for PACT
    else:
        return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    if isinstance(scale, torch.Tensor):
        return (input + zero_point.to(input.device)) / scale.to(input.device) # HACK for PACT
    else:
        return (input + zero_point) / scale


class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, num_bits_test, dequantize, inplace, mean_aligned=False):
        # pdb.set_trace()
        ctx.save_for_backward(input.detach().clone(), clip_val)
        m1 = input.detach().clone().mean()
        # if num_bits == 32:
        #     output = input
        # el
        if num_bits == 1:
            output = torch.sign(input)
        else:
            if inplace:
                ctx.mark_dirty(input)
            # scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, clip_val.data[0], signed=False)
            # output = clamp(input, 0, clip_val.data[0], inplace)
            scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, clip_val.data, signed=False)
            if isinstance(clip_val, torch.Tensor):
                if input.min() < 0:
                    raise ValueError('[JC] SENQNN: input to ClippedLinearQuantization should be non-negative.')
               # clip_val_map = torch.ones(input.shape).to(clip_val.device).mul(clip_val)
               # output = torch.where(input>clip_val_map, clip_val_map, input) # [JC] assume input >= 0
                output = torch.where(input>clip_val, torch.ones_like(input)*clip_val, input) ##naigang: to combine last two lines for speedup
            else:
                output = clamp(input, 0, clip_val.data, inplace)
            out_bit_h = linear_quantize(output, scale, zero_point, inplace)
            m2 = linear_dequantize(out_bit_h.detach().clone(), scale, zero_point, inplace).mean()
            if num_bits_test == -1:
                num_bits_test = num_bits
            d = int(2 ** (num_bits - num_bits_test))
            if d != 1:
                out_bit_l = (torch.floor(out_bit_h / d) * d).float()
            else:
                out_bit_l = out_bit_h
            if dequantize:
                output = linear_dequantize(out_bit_l, scale, zero_point, inplace)
                m3 = output.detach().clone().mean()
                if mean_aligned:
                    if m3 != 0:
                        output = output * m1 / m3
                    else:
                        output = output
                else:
                    if m3 != 0:
                        output = output * m2 / m3
                    else:
                        output = output
            else:
                output = out_bit_l
        # pdb.set_trace()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input[input.le(0)] = 0
       # grad_input[input.ge(clip_val.data)] = 0
        #Naigang: modify last two lines for speedup
       # grad_input = torch.where(input<=0, torch.zeros_like(grad_input), grad_input)
        grad_input = torch.where(input<0, torch.zeros_like(grad_input), grad_input)
       # grad_input = torch.where(input>=clip_val, torch.zeros_like(grad_input), grad_input)
        grad_input = torch.where(input>clip_val, torch.zeros_like(grad_input), grad_input)

        grad_alpha = grad_output.clone()
       # grad_alpha[input.lt(clip_val.data)] = 0
        grad_alpha = torch.where(input<clip_val, torch.zeros_like(grad_alpha), grad_alpha) #naigang: modify for speedup
      #  print("grad_alpha before sum {}".format(grad_alpha))
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
#        print("grad_alpha after sum {}".format(grad_alpha))
        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, None, None, None, None, None

class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, num_bits_test,  clip_val, dequantize=True, inplace=False, mean_aligned=False):
        """single sided original PACT"""
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.num_bits_test = num_bits_test
        self.clip_val = clip_val
        self.dequantize = dequantize
        self.inplace = inplace
        self.mean_aligned = mean_aligned
        # pdb.set_trace()

    def forward(self, input):
        input = LearnedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.num_bits, self.num_bits_test,
                                                      self.dequantize, self.inplace, self.mean_aligned)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val[0], inplace_str)

class LearnedClippedLinearQuantization2(nn.Module):
    def __init__(self, num_bits, num_bits_test, dequantize=True, inplace=False, mean_aligned=False):
        """single sided original PACT"""
        super(LearnedClippedLinearQuantization2, self).__init__()
        self.num_bits = num_bits
        self.num_bits_test = num_bits_test
        self.dequantize = dequantize
        self.inplace = inplace
        self.mean_aligned = mean_aligned
        # pdb.set_trace()

    def forward(self, input, clip_val):
        input = LearnedClippedLinearQuantizeSTE.apply(input, clip_val, self.num_bits, self.num_bits_test,
                                                      self.dequantize, self.inplace, self.mean_aligned)
        return input

    def __repr__(self):
        return '{0}(num_bits={1})'.format(self.__class__.__name__, self.num_bits)



def sawb_quantization_params(num_bits, out):
    with torch.no_grad():
        # pdb.set_trace()
        x = out.flatten()
        mu = x.abs().mean()
        std = x.mul(x).mean().sqrt()

       # dic_coeff = {2:(3.212, -2.178), 3:(7.509, -6.892), 4:(12.68, -12.80), 5:(17.74, -18.64)}
        dic_coeff = {2:(3.12, -2.064), 3:(7.509, -6.892), 4:(12.68, -12.80), 5:(17.74, -18.64)}
        if num_bits > 5:
            raise ValueError('SAWB not implemented for num_bits={}'.format(num_bits))
        coeff = dic_coeff[num_bits]
        clip_val = coeff[1] * mu + coeff[0] * std

        return clip_val

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, d, zero_point, dequantize, inplace, mean_aligned=False):
        if inplace:
            ctx.mark_dirty(input)
        m1 = input.detach().clone().mean()
        out_bit_h = linear_quantize(input, scale, zero_point, inplace)
        m2 = linear_dequantize(out_bit_h.detach().clone(), scale, zero_point, inplace).mean()
        # print(out_bit_h)
        if d != 1:
            out_bit_l = (torch.floor(out_bit_h / d) * d).float()
        else:
            out_bit_l = out_bit_h
        if dequantize:
            output = linear_dequantize(out_bit_l, scale, zero_point, inplace)
            m3 = output.detach().clone().mean()
            if mean_aligned:
                if m3 != 0:
                    output = output * m1 / m3
                else:
                    output = output
            else:
                if m3 != 0:
                    output = output * m2 / m3
                else:
                    output = output
        else:
            output = out_bit_l
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None, None, None


def sawb_quantize_param(out, num_bits):
    dequantize=True
    inplace=False
    scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, 1, signed=False)
    clip_val = sawb_quantization_params(num_bits, out)
    out = out.mul(1/clip_val).clamp(-1, 1).mul(0.5).add(0.5)
    out = LinearQuantizeSTE.apply(out, scale, zero_point, dequantize, inplace)
    out = (2 * out - 1) * clip_val
    return out


def dorefa_quantize_param(out, num_bits, num_bits_test, mean_aligned=False):
    dequantize=True
    inplace=False
    # if num_bits == 32:
    #     out = out
    # el
    if num_bits == 1:
        out = torch.sign(out)
    else:
        scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, 1, signed=False)
        out = out.tanh()
        out = out / (2 * out.abs().max()) + 0.5
        if num_bits_test == -1:
            num_bits_test = num_bits
        assert (num_bits_test <= num_bits)
        d = int(2 ** (num_bits - num_bits_test))
        out = LinearQuantizeSTE.apply(out, scale, d,  zero_point, dequantize, inplace, mean_aligned)
        out = 2 * out - 1
    return out

def conv2d_Q_fn(w_bit, w_bit_test=-1, mean_aligned=False):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = sawb_quantize_param

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight,  self.w_bit)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q

def conv2d_Q_fn_dorefa(w_bit, w_bit_test=-1, mean_aligned=False):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.w_bit_test = w_bit_test
      self.mean_aligned = mean_aligned
      self.quantize_fn = dorefa_quantize_param

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight,  self.w_bit, self.w_bit_test, self.mean_aligned)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def conv2d_Q_fn_dorefa2(mean_aligned=False):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
        self.quantize_fn = dorefa_quantize_param
        self.mean_aligned = mean_aligned

    def forward(self, input,  w_bit, w_bit_test=-1, order=None):
      weight_q = self.quantize_fn(self.weight, w_bit, w_bit_test, self.mean_aligned)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q

def activation_quantize_fn(clip_val, a_bit, a_bit_test=-1, mean_aligned=False):
    return LearnedClippedLinearQuantization(a_bit, a_bit_test, clip_val=clip_val, dequantize=True, inplace=False,
                                            mean_aligned=mean_aligned)


def activation_quantize_fn2(a_bit, a_bit_test=-1, mean_aligned=False):
    return LearnedClippedLinearQuantization2(a_bit, a_bit_test, dequantize=True, inplace=False,
                                            mean_aligned=mean_aligned)


if __name__ == '__main__':
  ones = 0.0001 * torch.arange(0, 10000).float()
  import pdb
  pdb.set_trace()
  # q1 = modified_uniform_quantize(2)(ones)
  # q2 = modified_uniform_quantize(3)(ones)
  # q3 = modified_uniform_quantize(4)(ones)
  q1 = dorefa_quantize_param(ones, 4, 2, False)
  q2 = dorefa_quantize_param(ones, 4, 3, False)
  q3 = dorefa_quantize_param(ones, 4, 4, False)
  import pdb
  pdb.set_trace()
  q1_np = q1.numpy()
  q2_np = q2.numpy()
  q3_np = q3.numpy()
  import matplotlib.pyplot as plt
  import numpy as np
  plt.plot(np.arange(0, 10000) * 0.0001, q1_np)
  plt.plot(np.arange(0, 10000) * 0.0001, q2_np)
  plt.plot(np.arange(0, 10000) * 0.0001, q3_np)
  plt.grid()
  plt.legend(['2 bit', '3 bit', '4 bit'])
  plt.savefig('naigang_quant2.png')
