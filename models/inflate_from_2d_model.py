import torch
from collections import OrderedDict

def inflate_from_2d_model(state_dict_2d, state_dict_3d, skipped_keys=None, inflated_dim=2):

    if skipped_keys is None:
        skipped_keys = []

    missed_keys = []
    new_keys = []
    for old_key in state_dict_2d.keys():
        if old_key not in state_dict_3d.keys():
            missed_keys.append(old_key)
    for new_key in state_dict_3d.keys():
        if new_key not in state_dict_2d.keys():
            tokens = new_key.split('.')
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
            if new_k not in state_dict_2d.keys():
                new_keys.append(new_key)
    print("Missed tensors: {}".format(missed_keys))
    print("New tensors: {}".format(new_keys))
    print("Following layers will be skipped: {}".format(skipped_keys))

    state_d = OrderedDict()
    unused_layers = [k for k in state_dict_2d.keys()]
    uninitialized_layers = [k for k in state_dict_3d.keys()]
    initialized_layers = []
    for key, _ in state_dict_3d.items():
        skipped = False
        for skipped_key in skipped_keys:
            if skipped_key in key:
                skipped = True
                break
        if skipped:
            continue


        exist = False
        if key in state_dict_2d.keys():
            new_key = key
            exist = True
        else:
            tokens = key.split('.')
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
            if new_k in state_dict_2d.keys():
                new_key = new_k
                exist = True

        if exist:
            # TODO: a better way to identify conv layer?
            # if 'conv.weight' in key or \
            #         'conv1.weight' in key or 'conv2.weight' in key or 'conv3.weight' in key or \
            #         'downsample.0.weight' in key:
            value = state_dict_2d[new_key]
            new_value = value
            if value.ndimension() == 4 and 'weight' in key:
                value = torch.unsqueeze(value, inflated_dim)
                # value.unsqueeze_(inflated_dim)
                repeated_dim = torch.ones(state_dict_3d[key].ndimension(), dtype=torch.int)
                repeated_dim[inflated_dim] = state_dict_3d[key].size(inflated_dim)
                new_value = value.repeat(repeated_dim.tolist())
            state_d[key] = new_value
            initialized_layers.append(key)
            uninitialized_layers.remove(key)
            if new_key in unused_layers:
                unused_layers.remove(new_key)

    print("Initialized layers: {}".format(initialized_layers))
    print("Uninitialized layers: {}".format(uninitialized_layers))
    print("Unused layers: {}".format(unused_layers))

    return state_d


def convert_rgb_model_to_others(state_dict, input_channels, ks=7):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "conv1.weight" in key:
            o_c, in_c, k_h, k_w = value.shape
        else:
            o_c, in_c, k_h, k_w = 0, 0, 0, 0
        if in_c == 3 and k_h == ks and k_w == ks:
            # average the weights and expand to all channels
            new_shape = (o_c, input_channels, k_h, k_w)
            new_value = value.mean(dim=1, keepdim=True).expand(new_shape).contiguous()
        else:
            new_value = value
        new_state_dict[key] = new_value
    return new_state_dict

def convert_rgb_model_to_group(src_state_dict, target_state_dict, groups):
    new_state_dict = {}
    for key, value in target_state_dict.items():
        if key in src_state_dict:
            if len(src_state_dict[key].shape) == 0: #skip non-parameters
                new_state_dict[key] = src_state_dict[key]
                #print ('NO DATA === %s' % (key))
                continue
            #print (key, target_state_dict[key].shape, src_state_dict[key].shape)
            assert target_state_dict[key].shape[0] == groups * src_state_dict[key].shape[0]
            assert len(src_state_dict[key].shape) == 1 or len(src_state_dict[key].shape) == 4
            #new_state_dict[key] = src_state_dict[key]
            if len(src_state_dict[key].shape) == 1:
                new_state_dict[key] = src_state_dict[key].repeat(groups)
            else:
                new_state_dict[key] = src_state_dict[key].repeat(groups, 1, 1, 1)
            #print (value.shape, src_state_dict[key].shape)
        #else:
            #print ('NOT COPIED ***** %s' % (key))
    return new_state_dict
