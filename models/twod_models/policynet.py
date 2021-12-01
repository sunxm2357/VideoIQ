"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import sys
sys.path.insert(0, '../../')
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from models.twod_models.ops.imagenet_pact import conv2d_Q_fn_dorefa, activation_quantize_fn2
from models.twod_models.ops.twoside_pact import LearnedTwosidedClippedLinearQuantization


__all__ = ['mobilenetv2']


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


def init_hidden(batch_size, cell_size):
    init_cell = torch.zeros(batch_size, cell_size)
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, bit, bit_test=-1, mean_aligned=False):
    conv_fn = conv2d_Q_fn_dorefa(bit, bit_test, mean_aligned)
    return nn.Sequential(
        conv_fn(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, bit, bit_test=-1, mean_aligned=False):
    conv_fn = conv2d_Q_fn_dorefa(bit, bit_test, mean_aligned)
    return nn.Sequential(
        conv_fn(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, bit, q_init, mean_aligned=False):
        super(InvertedResidual, self).__init__()
        self.bit = bit
        if bit == 32:
            conv_fn = nn.Conv2d
        else:
            conv_fn = conv2d_Q_fn_dorefa(bit, -1, mean_aligned)
            self.act_q = activation_quantize_fn2(bit, -1, mean_aligned)

        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        if expand_ratio == 1:
            if bit != 32:
                self.register_parameter('q_alpha_1', nn.Parameter(q_init * torch.ones(1), requires_grad=True))
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_fn(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if bit != 32:
                self.register_parameter('q_alpha_1', nn.Parameter(q_init * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha_2', nn.Parameter(q_init * torch.ones(1), requires_grad=True))
                self.register_parameter('q_alpha_3', nn.Parameter(q_init * torch.ones(1), requires_grad=True))
                self.act_q2 = LearnedTwosidedClippedLinearQuantization(bit, self.q_alpha1, self.q_alpha2)

            self.conv = nn.Sequential(
                # pw
                conv_fn(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv_fn(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def conv_q(self, x):
        if self.expand_ratio == 1:
            x = self.conv[2](self.conv[1](self.conv[0](x)))
            alpha = getattr(self, 'q_alpha_1')
            x = self.act_q(x, alpha)
            x = self.conv[4](self.conv[3](x))
        else:
            x = self.act_q2(x)
            x = self.conv[2](self.conv[1](self.conv[0](x)))
            x = self.conv[5](self.conv[4](self.conv[3](x)))
            alpha = getattr(self, 'q_alpha_3')
            x = self.act_q(x, alpha)
            x = self.conv[7](self.conv[6](x))
        return x

    def forward(self, x):
        if self.bit == 32:
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)
        else:
            if self.identity:
                return x + self.conv_q(x)
            else:
                return self.conv_q(x)


# TODO (ximeng): implement the quantized version
class MobileNetV2(nn.Module):
    def __init__(self, num_frames, prec_dim, hidden_dim, dropout, skip_list, bit, q_init, is_policy_pred=False, num_classes=1000, mean_aligned=False, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.register_buffer('temperature', torch.ones(1) * 5.0)
        self.skip_list = skip_list
        self.orig_num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.is_policy_pred = is_policy_pred
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, 32)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, bit, q_init,
                                    mean_aligned=mean_aligned))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, bit, mean_aligned=mean_aligned)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.LSTMCell(input_size=1280, hidden_size=self.hidden_dim, bias=True)

        self.prec_dim = prec_dim
        self.skip_dim = len(self.skip_list)
        self.action_dim = self.prec_dim + self.skip_dim

        self.linear = nn.Linear(self.hidden_dim, self.action_dim)
        if self.is_policy_pred:
            self.fc = nn.Linear(self.hidden_dim, num_classes)

        self._initialize_weights()

    def set_temperature(self, temperature):
        self.temperature = temperature

    def decay_temperature(self, decay_ratio=None):
        if decay_ratio:
            self.temperature *= decay_ratio
        print("Current temperature: {}".format(self.temperature), flush=True)

    @staticmethod
    def get_deterministic_decision(logits):
        decision_dim = logits.shape[1]
        p_t_max = torch.argmax(logits, dim=1)
        r_t = one_hot(p_t_max, list(range(decision_dim)))
        return r_t

    def lstm_forward(self, feat_lite, deterministic=False):
        batch_size = feat_lite.shape[0]
        device = feat_lite.get_device()

        remain_skip_vector = torch.zeros(batch_size, 1)
        old_hx = None
        old_r_t = None

        hx = init_hidden(batch_size, self.hidden_dim)
        cx = init_hidden(batch_size, self.hidden_dim)

        r_list = []
        feat_list = []
        for t in range(self.orig_num_frames):
            hx, cx = self.rnn(feat_lite[:, t], (hx, cx))
            feat_t = hx
            feat_list.append(feat_t)
            p_t = torch.log(F.softmax(self.linear(feat_t), dim=1).clamp(min=1e-8))

            if deterministic:
                r_t = self.get_deterministic_decision(p_t)
            else:
                r_t = torch.cat([F.gumbel_softmax(p_t[b_i:b_i + 1], self.temperature, True) for b_i in range(p_t.shape[0])])

            if old_hx is not None:
                take_bool = remain_skip_vector > 0.5
                if device != -1:
                    take_old = torch.tensor(take_bool, dtype=torch.float, device=device)
                    take_curr = torch.tensor(~take_bool, dtype=torch.float, device=device)
                else:
                    take_old = torch.tensor(take_bool, dtype=torch.float)
                    take_curr = torch.tensor(~take_bool, dtype=torch.float)

                hx = old_hx * take_old + hx * take_curr
                r_t = old_r_t * take_old + r_t * take_curr

            for batch_i in range(batch_size):
                for skip_i in range(self.skip_dim):
                    if remain_skip_vector[batch_i][0] < 0.5 and r_t[batch_i][self.prec_dim + skip_i] > 0.5:
                        remain_skip_vector[batch_i][0] = self.skip_list[skip_i]

            old_hx = hx
            old_r_t = r_t
            r_list.append(r_t)
            remain_skip_vector = (remain_skip_vector - 1).clamp(0)

        if self.is_policy_pred:
            feat = torch.stack(feat_list, dim=1).mean(dim=1)
            output = self.fc(feat)
        else:
            output = None

        return torch.stack(r_list, dim=1), output


    def forward(self, x, deterministic=False):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        feat_lite = x.view(batch_size, self.orig_num_frames, -1)
        device = feat_lite.get_device()

        remain_skip_vector = torch.zeros(batch_size, 1)
        old_hx = None
        old_r_t = None
        old_p_t = None

        hx = init_hidden(batch_size, self.hidden_dim)
        cx = init_hidden(batch_size, self.hidden_dim)

        r_list = []
        p_list = []
        feat_list = []
        for t in range(self.orig_num_frames):
            hx, cx = self.rnn(feat_lite[:, t], (hx, cx))
            feat_t = hx
            feat_list.append(feat_t)
            p_t = torch.log(F.softmax(self.linear(feat_t), dim=1).clamp(min=1e-8))

            if deterministic:
                r_t = self.get_deterministic_decision(p_t)
            else:
                r_t = torch.cat([F.gumbel_softmax(p_t[b_i:b_i + 1], self.temperature, True) for b_i in range(p_t.shape[0])])

            if old_hx is not None:
                take_bool = remain_skip_vector > 0.5
                if device != -1:
                    take_old = torch.tensor(take_bool, dtype=torch.float, device=device)
                    take_curr = torch.tensor(~take_bool, dtype=torch.float, device=device)
                else:
                    take_old = torch.tensor(take_bool, dtype=torch.float)
                    take_curr = torch.tensor(~take_bool, dtype=torch.float)

                hx = old_hx * take_old + hx * take_curr
                r_t = old_r_t * take_old + r_t * take_curr
                p_t = old_p_t * take_old + p_t * take_curr

            for batch_i in range(batch_size):
                for skip_i in range(self.skip_dim):
                    if remain_skip_vector[batch_i][0] < 0.5 and r_t[batch_i][self.prec_dim + skip_i] > 0.5:
                        remain_skip_vector[batch_i][0] = self.skip_list[skip_i]

            old_hx = hx
            old_r_t = r_t
            old_p_t = p_t

            r_list.append(r_t)
            p_list.append(p_t)
            remain_skip_vector = (remain_skip_vector - 1).clamp(0)

        if self.is_policy_pred:
            feat = torch.stack(feat_list, dim=1).mean(dim=1)
            output = self.fc(feat)
        else:
            output = None

        return torch.stack(r_list, dim=1), torch.stack(p_list, dim=1), output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)


if __name__ == '__main__':
    from torchsummary import torchsummary

    model = mobilenetv2(num_frames=16, prec_dim=3, hidden_dim=512, dropout=0.8, skip_list=[1, 2, 4], bit=4, q_init=10,
                        mean_aligned=False, width_mult=1.)

    dummy_data = (24, 84, 84)
    model.eval()
    model_summary = torchsummary.summary(model, input_size=dummy_data)
    # print(model)
    # print(model_summary)
    # from utils.utils import compute_flops
    #
    # # compute flops for the entire policynet
    # dummy_input = torch.ones(1, 24, 84, 84)
    # flops_all = compute_flops(model, dummy_input, False, {})
    # print("FLOPs of PolicyNet", flops_all)
    #
    # # compute first layer of the policynet
    # dummy_input = torch.ones(8, 3, 84, 84)
    # flops_1st = compute_flops(model.features[0], dummy_input, False, {})
    # print("FLOPs of 1st layer in PolicyNet", flops_1st)
    #
    # # compute last layer of the policynet
    # dummy_input = torch.ones(1, 1280)
    # flops_last = compute_flops(model.rnn, dummy_input, False, {})
    # print("FLOPs of lstm in PolicyNet", flops_last)
    #
    # bit = 8
    # bitops = (flops_all - flops_1st * 8) * bit * bit + flops_1st * 8 * 32 * 32
    # print("BitOps", bitops)
    #
    # bit = 32
    # bitops = (flops_all - flops_1st * 8) * bit * bit + flops_1st * 8 * 32 * 32
    # print("BitOps", bitops)