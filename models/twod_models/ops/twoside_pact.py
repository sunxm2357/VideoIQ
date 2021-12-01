import torch
import torch.nn as nn


def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val)
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out


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
        return torch.round(scale.to(input.device) * input - zero_point.to(input.device)) # HACK for PACT
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



class LearnedTwosidedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, clip_valn, num_bits, num_bits_test, dequantize, inplace, mean_aligned=False):
        # pdb.set_trace()
        ctx.save_for_backward(input, clip_val, clip_valn)
        m1 = input.detach().clone().mean()
        if num_bits == 32:
            output = input
        elif num_bits == 1:
            output = torch.sign(input)
        else:
            if inplace:
                ctx.mark_dirty(input)
            scale, zero_point = asymmetric_linear_quantization_params(num_bits, clip_valn.data, clip_val.data, integral_zero_point=False, signed=False)
            if isinstance(clip_val, torch.Tensor):
                # if input.min() < 0:
                #     raise ValueError('[JC] SENQNN: input to ClippedLinearQuantization should be non-negative.')
                # if clip_val.max() < 0:
                #     raise ValueError('[JC] SENQNN: clip_val to LearnedTwosidedClippedLinearQuantizeSTE should be non-negative.')
                # if clip_valn.min() > 0:
                #     raise ValueError('[JC] SENQNN: clip_valn={} to LearnedTwosidedClippedLinearQuantizeSTE should be non-positive.'.format(clip_valn.min()))
               # if clip_val.min() < clip_valn.max():
               #     raise ValueError('[JC] SENQNN: in LearnedTwosidedClippedLinearQuantizeSTE, clip_val.min={} should be larger than clip_valn.max={}.'.format(clip_val.min(), clip_valn.max()))
                # pdb.set_trace()
                #clip_val_map = torch.ones(input.shape).to(clip_val.device).mul(clip_val)
                #output = torch.where(input>clip_val_map, clip_val_map, input)
                output = torch.where(input>clip_val, torch.ones_like(input)*clip_val, input) ##naigang: to combine last two lines for speedup
                #clip_valn_map = torch.ones(input.shape).to(clip_valn.device).mul(clip_valn)
                #output = torch.where(output<clip_valn_map, clip_valn_map, output)
                output = torch.where(output<clip_valn, torch.ones_like(input)*clip_valn, output) ##naigang: to combine last two lines for speedup
                # pdb.set_trace()
            else:
                output = clamp(input, clip_valn.data, clip_val.data, inplace)
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
        input, clip_val, clip_valn = ctx.saved_tensors
        grad_input = grad_output.clone()
       # grad_input[input.le(clip_valn.data)] = 0
       # grad_input[input.ge(clip_val.data)] = 0
        #Naigang: modify last two lines for speedup
        grad_input = torch.where(input<=clip_valn, torch.zeros_like(grad_input), grad_input)
        grad_input = torch.where(input>=clip_val, torch.zeros_like(grad_input), grad_input)


        grad_alpha = grad_output.clone()
        #grad_alpha[input.lt(clip_val.data)] = 0
        grad_alpha = torch.where(input<clip_val, torch.zeros_like(grad_alpha), grad_alpha) #naigang: modify for speedup
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        grad_alphan = grad_output.clone()
        #grad_alphan[input.gt(clip_valn.data)] = 0
        grad_alphan = torch.where(input>clip_valn, torch.zeros_like(grad_alphan), grad_alphan) #naigang: modify for speedup
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)

        # pdb.set_trace()


        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, grad_alphan, None, None, None, None, None


class LearnedTwosidedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, clip_val, clip_valn, num_bits_test=-1, dequantize=True, inplace=False, mean_aligned=False):
        """two-sided original PACT"""
        super(LearnedTwosidedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.num_bits_test = num_bits_test
        self.dequantize = dequantize
        self.inplace = inplace
        self.clip_val = clip_val
        self.clip_valn = clip_valn
        self.mean_aligned = mean_aligned

    def forward(self, input):
        input = LearnedTwosidedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.clip_valn, self.num_bits, self.num_bits_test,
                                                              self.dequantize, self.inplace, self.mean_aligned)
        return input

    def __repr__(self):
        clip_str = ', pos-clip={}, neg-clip={}'.format(self.clip_val[0], self.clip_valn[0])
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}{2}{3})'.format(self.__class__.__name__, self.num_bits, clip_str, inplace_str)


class LearnedTwosidedClippedLinearQuantization2(nn.Module):
    def __init__(self, num_bits, num_bits_test=-1, dequantize=True, inplace=False, mean_aligned=False):
        """two-sided original PACT"""
        super(LearnedTwosidedClippedLinearQuantization2, self).__init__()
        self.num_bits = num_bits
        self.num_bits_test = num_bits_test
        self.dequantize = dequantize
        self.inplace = inplace
        self.mean_aligned = mean_aligned

    def forward(self, input, clip_val, clip_valn):
        input = LearnedTwosidedClippedLinearQuantizeSTE.apply(input, clip_val, clip_valn, self.num_bits, self.num_bits_test,
                                                              self.dequantize, self.inplace, self.mean_aligned)
        return input

    def __repr__(self):
        clip_str = ', '
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}{2}{3})'.format(self.__class__.__name__, self.num_bits, clip_str, inplace_str)
