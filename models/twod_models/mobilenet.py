import torch
from torch import nn
from functools import partial
import torch.utils.model_zoo as model_zoo
from functools import partial
from inspect import signature
#from .utils import load_state_dict_from_url
from models.twod_models.temporal_modeling import temporal_modeling_module


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, num_frames, stride, expand_ratio, temporal_module = None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.tam = temporal_module(duration=num_frames, channels=inp) \
            if temporal_module is not None else None


    def forward(self, x):
        if self.use_res_connect:
            identity = x
            if self.tam:
                x = self.tam(x)
            x = self.conv(x)
            return identity + x
           # return x + self.conv(x)
        else:
            if self.tam:
                x = self.tam(x)
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_frames,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None, 
                 dropout=0.5,
                 pooling_method='max',
                 without_t_stride=False, 
                 temporal_module=None
                ):
        """
        MobileNet V2 main class
        Args:
            num_frames (int): Number of frames
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.orig_num_frames = num_frames
        self.num_frames = num_frames
        self.pooling_method = pooling_method.lower()
        self.temporal_module = temporal_module
        self.width_mult = width_mult
        self.without_t_stride = without_t_stride

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, num_frames, stride, expand_ratio=t, temporal_module=temporal_module))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.last_channel, num_classes)

        # building classifier
        #self.action_classifier = nn.Sequential(
        #    nn.Dropout(dropout),
        #    nn.Linear(self.last_channel, num_classes),
        #)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def extract_feature(self, x):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]

        return x

    def _forward_impl(self, x):
        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        n_t, c = x.shape
        out = x.view(batch_size, -1, c)

        # average the prediction from all frames
        out = torch.mean(out, dim=1)

        #x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        #x = self.classifier(x)
        return out

    def forward(self, x):
        return self._forward_impl(x)

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
        name += 'mobilenet_v2-{}'.format(int(self.width_mult*100))
        if not self.without_t_stride:
            name += "-ts-{}".format(self.pooling_method)

        return name


def mobilenet_v2(width_mult, num_classes, without_t_stride, groups, temporal_module_name,
           dw_conv, blending_frames, blending_method, dropout, input_channels, imagenet_pretrained=True, **kwargs):

    temporal_module = None
    if temporal_module_name:
        temporal_module = partial(temporal_modeling_module, name=temporal_module_name,
                                          dw_conv=dw_conv, blending_frames=blending_frames,
                                          blending_method=blending_method)

    model = MobileNetV2(num_frames=groups, 
                        num_classes=num_classes,
                        width_mult=width_mult,
                        inverted_residual_setting = None,
                        round_nearest = 8,
                        block = None,
                        dropout = dropout,
                        without_t_stride = False,
                        temporal_module=temporal_module)

#    for key, value in model.state_dict().items():
#        if key == 'features.1.conv.0.0.weight':
#            print (value[0:2,...])

    if imagenet_pretrained:
        #state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], map_location='cpu')
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'], map_location='cpu')
#        state_dict.pop('fc.weight', None)
#        state_dict.pop('fc.bias', None)
        if input_channels != 3:  # convert the RGB model to others, like flow
            state_dict = convert_rgb_model_to_others(state_dict, input_channels, 7)
        model.load_state_dict(state_dict, strict=False)
    
#    for key, value in model.state_dict().items():
#        if key == 'features.1.conv.0.0.weight':
#            print (value[0:2,...])

   # for name, param in model.named_parameters():
   #     print (name, param.data.shape)
    #for key, value in state_dict.items():
    #    print (key, value)
    return model

'''
def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
'''
