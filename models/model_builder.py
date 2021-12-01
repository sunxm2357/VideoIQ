from . import (resnet, resnet_pact_ap, mobilenet_v2, resnet_videoiq)

MODEL_TABLE = {
    'resnet': resnet,
    'resnet_pact_ap': resnet_pact_ap,
    'resnet_videoiq': resnet_videoiq,
    'mobilenet_v2': mobilenet_v2,
}


def build_model(args):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    model = MODEL_TABLE[args.backbone_net](**vars(args))
    network_name = model.network_name if hasattr(model, 'network_name') else args.backbone_net
    #print (hasattr(model, 'network_name'), network_name)
    arch_name = "{dataset}-{modality}-{arch_name}".format(
        dataset=args.dataset, modality=args.modality, arch_name=network_name)
    arch_name += "-f{}".format(args.groups)

    # add setting info only in training

    arch_name += "-{}{}-bs{}{}-e{}".format(args.lr_scheduler, "-syncbn" if args.sync_bn else "",
                                         args.batch_size, '-' + args.prefix if args.prefix else "", args.epochs)
    return model, arch_name
