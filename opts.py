import argparse
from models.model_builder import MODEL_TABLE
from utils.dataset_config import DATASET_CONFIG


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    # adaptive model params
    parser.add_argument('--bit_width_family', default=[8, 4], type=int, metavar='N', nargs="+",
                        help='bit width family of weights in uniform quantization')
    parser.add_argument('--skip_list', default=[], type=int, metavar='N', nargs="+",
                        help='the skipping choices')
    parser.add_argument('--p_lr', '--p_learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--p_b_lr', '--p_backbone_learning-rate', default=0.00001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--efficient', action='store_true',
                        help='specify if using the efficient loss for the policy')
    parser.add_argument('--entropy', action='store_true',
                        help='specify if minimizing entropy of the policy')
    parser.add_argument('--balanced', action='store_true',
                        help='specify if using the balanced loss for the policy')
    parser.add_argument('--efficient_w', default=0.001, type=float,
                        help='the weight of efficient loss of the policy')
    parser.add_argument('--entropy_w', default=0.001, type=float,
                        help='the weight of entropy of the policy')
    parser.add_argument('--balanced_w', default=0.001, type=float,
                        help='the weight of balanced loss of the policy')
    parser.add_argument('--temperature_decay_rate', default=0.956, type=float,
                        help='decay rate of temperature every epoch')
    parser.add_argument('--resnet_imagenet_path', dest='resnet_imagenet_path', type=str,
                        help='the path of imagenet pretrained weights if not loading official weights')
    parser.add_argument('--resnet_imagenet_path_32fp', dest='resnet_imagenet_path_32fp', type=str,
                        help='the path of imagenet pretrained weights if not loading official weights')
    parser.add_argument('--resnet_imagenet_paths', dest='resnet_imagenet_paths', type=str, nargs="+",
                        help='the path of imagenet pretrained weights if not loading official weights')

    parser.add_argument('--mobilenet_imagenet_path', dest='mobilenet_imagenet_path', type=str,
                        help='the path of imagenet pretrained weights if not loading official weights')
    parser.add_argument('--p_epochs', default=50, type=int, metavar='N',
                        help='number of epochs to train the policy')
    parser.add_argument('--ap_epochs', default=80, type=int, metavar='N',
                        help='number of epochs to train ap in the joint training')
    parser.add_argument('--joint_epochs', default=50, type=int, metavar='N',
                        help='number of epochs to jointly train the policy and the backbone')
    parser.add_argument('--policy_epochs', default=20, type=int, metavar='N',
                        help='number of epochs to train the policy in the joint training')
    parser.add_argument('--f_epochs', default=50, type=int, metavar='N',
                        help='number of epochs to finetune the backbone')
    parser.add_argument('--rand_policy', action='store_true',
                        help='specify if using the random policy during the test')
    parser.add_argument('--dist',  default=None, type=float, metavar='N', nargs="+",
                        help='the distribution while using the random policy during the test')

    parser.add_argument('--is_policy_pred', action='store_true', help='specify if use prediction supervision for the'
                                                                      ' policy network')

    parser.add_argument('--deterministic', action='store_true', help='specify if use the deterministic policy')

    parser.add_argument('--use_fp_as_bb', action='store_true', help='specify if use 32 full precision feature as the'
                                                                      ' backbone network')

    parser.add_argument('--policy_w', default=1, type=float,
                        help='the weight of prediction loss of the policy network')

    parser.add_argument('--skipped_keys', default=[], type=str, metavar='N', nargs="+",
                        help='specify if skipped keys are needed in the model loading')

    parser.add_argument('--threed_model_pretrained', action='store_true', help='specify if the pretrained model is 3d model')


    # ap model params
    parser.add_argument('--progressive2_type', default='avg', type=str, metavar='N', choices=['avg', 'learnable'],
                        help='the combination type of linear progressive')
    parser.add_argument('--is_32fp', action='store_true', help='specify if use full precision in KD')
    parser.add_argument('--ps_w', default=[1.0], type=float, metavar='N', nargs="+",
                        help='the precision-specific weighting')

    # debug
    parser.add_argument('--debug', action='store_true',
                        help='specify if swapping during the training')


    # swap
    parser.add_argument('--stochastic_depth', action='store_true',
                        help='combine stochastic depth with interactive swapping')
    parser.add_argument('--swap', action='store_true',
                        help='specify if swapping during the training')
    parser.add_argument('--swapping_mode', type=str, help='the mode of swapping in the any precision network')
    parser.add_argument('--interactive_w', type=float, help='the weight of tensor difference in the interactive swapping')

    parser.add_argument('--order', default='descending', type=str, help='the order of the stochastic depth',
                        choices=['ascending', 'descending'])

    parser.add_argument('--progressive_train', action='store_true', help='use progressive training strategy')
    parser.add_argument('--progressive_start', default=20, type=int, help='the start epoch of using KD loss if using swap')
    parser.add_argument('--progressive_end', default=90, type=int, help='the end epoch of using KD loss if using swap')

    # efficient_net
    parser.add_argument('--use_relu', action='store_true',
                        help='specify if using relu instead of swish in the efficient net')

    # model definition
    parser.add_argument('--backbone_net', default='s3d', type=str, help='backbone network',
                        choices=list(MODEL_TABLE.keys()))
    parser.add_argument('-d', '--depth', default=18, type=int, metavar='N',
                        help='depth of resnet (default: 18)', choices=[18, 34, 50, 101, 152])
    parser.add_argument('--w_bits', default=[4], type=int, metavar='N', nargs="+",
                        help='bit width of weights in uniform quantization')
    parser.add_argument('--a_bits', default=[4], type=int, metavar='N', nargs="+",
                        help='bit width of activations in uniform quantization')
    parser.add_argument('--w_bit_width_family', default=[8, 4], type=int, metavar='N', nargs="+",
                        help='bit width family of weights in uniform quantization')
    parser.add_argument('--a_bit_width_family', default=[8, 4], type=int, metavar='N', nargs="+",
                        help='bit width family of activations in uniform quantization')
    parser.add_argument('--switch_bn', action='store_true',
                        help='switch Batch Norm for different bit widths')
    parser.add_argument('--switch_clipval', action='store_true',
                        help='switch clipping values for different bit widths')
    parser.add_argument('--w_bit_test', default=-1, type=int, metavar='N',
                        help='bit width of weights in uniform quantization during test '
                             '(default: -1, the same as the training)')
    parser.add_argument('--a_bit_test', default=-1, type=int, metavar='N',
                        help='bit width of activations in uniform quantization '
                             '(default: -1, the same as the training)')
    parser.add_argument('--mean_aligned', action='store_true',
                        help='align the mean of tensors with the full precision model')
    parser.add_argument('--q_alpha', default=1, type=float, metavar='N',
                        help='fixed clip level in quantization')
    parser.add_argument('--q_init', default=2, type=int, metavar='N', nargs="+",
                        help='the initial value of clip level in pact')
    parser.add_argument('--p_bit', default=32, type=int, metavar='N',
                        help='the bit width of the policy net')
    parser.add_argument('--loss_type', default='CE', type=str, metavar='N', choices=['CE', 'KD', 'KD_CE'],
                        help='the loss type for Any Precision Network')
    parser.add_argument('--review_epochs', default=[0], type=int, metavar='N', nargs="+",
                        help='the start epoch of each review')
    parser.add_argument('--p_start', default=0.1, type=float, metavar='N', help='the start of swapping probability')
    parser.add_argument('--kd_weight', default=1, type=float, metavar='N', help='the loss weight for KD loss')
    parser.add_argument('--server', default='aimos', type=str, metavar='N', help='the server name')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout ratio before the final layer')
    parser.add_argument('--groups', default=16, type=int, help='number of frames')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency')
    parser.add_argument('--alpha', default=2, type=int, metavar='N',
                        help='[blresnet] ratio of channels')
    parser.add_argument('--beta', default=4, type=int, metavar='N',
                        help='[blresnet] ratio of layers')
    parser.add_argument('--fpn_dim', type=int, default=-1, help='enable 3D fpn on 2d net')
    parser.add_argument('--without_t_stride', dest='without_t_stride', action='store_true',
                        help='skip the temporal stride in the model')
    parser.add_argument('--pooling_method', default='max',
                        choices=['avg', 'max'], help='method for temporal pooling method or '
                                                     'which pool3d module')
    parser.add_argument('--dw_t_conv', dest='dw_t_conv', action='store_true',
                        help='[S3D model] enable depth-wise conv for temporal modeling')
    parser.add_argument('--prefix', default='', type=str, help='model prefix')
    # model definition: temporal model for 2D models
    parser.add_argument('--temporal_module_name', default=None, type=str,
                        help='which temporal aggregation module to use, '
                             'TSM, TAM and NonLocal are only available on 2D model',
                        choices=[None, 'TSM', 'TAM', 'MAX', 'AVG', 'Channel-Max', 'NonLocal', 'GroupTAM', 'GroupExpTAM',
                                 'TSM+NonLocal', 'TAM+NonLocal'])
    parser.add_argument('--blending_frames', default=3, type=int)
    parser.add_argument('--blending_method', default='sum',
                        choices=['sum', 'max', 'maxnorm'], help='method for blending channels')
    parser.add_argument('--no_dw_conv', dest='dw_conv', action='store_false',
                        help='[2D model] disable depth-wise conv for temporal modeling')
    parser.add_argument('--consensus', default='avg', type=str, help='which consesus to use',
                        choices=['avg', 'trn', 'multitrn'])
    parser.add_argument('--width_mult', default=1.0, type=float)
    parser.add_argument('--tam_pos', default='all', type=str, help='which consesus to use',
                        choices=['all', 'half', 'first', 'last', 'half_2'])

    # training setting
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--disable_cudnn_benchmark', dest='cudnn_benchmark', action='store_false',
                        help='Disable cudnn to search the best mode (avoid OOM)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--q_lr', '--q_learning-rate', default=[0.1], type=float, nargs="+",
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau'])
    parser.add_argument('--lr_steps', default=[15, 30, 45], type=float, nargs="+",
                        metavar='LRSteps', help='[step]: use a single value: the periodto decay '
                                                'learning rate by 10. '
                                                '[multisteps] epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='enable nesterov momentum optimizer')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--q_weight-decay', '--q_wd', default=[5e-4], type=float, nargs="+",
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--auto_resume', action='store_true', help='if the log folder includes a checkpoint, automatically resume')
    parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--backbone_pretrained', dest='backbone_pretrained', type=str, metavar='PATH',
                        help='use pre-trained model for the backbone')
    parser.add_argument('--policy_pretrained', dest='policy_pretrained', type=str, metavar='PATH',
                        help='use pre-trained model for the policy net')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
                        help='clip the total norm of gradient before update parameter')
    parser.add_argument('--label_smoothing', default=0.0, type=float, metavar='SMOOTHING',
                        help='label_smoothing against the cross entropy')
    parser.add_argument('--no_imagenet_pretrained', dest='imagenet_pretrained',
                        action='store_false',
                        help='disable to load imagenet pretrained model')
    parser.add_argument('--imagenet_path', dest='imagenet_path', type=str,
                        help='the path of imagenet pretrained weights if not loading official weights')
    parser.add_argument('--transfer', action='store_true',
                        help='perform transfer learning, remove weights in the fc '
                             'layer or the original model.')
    parser.add_argument('--bn_calibrate', action='store_true',
                        help='perform BatchNorm Calibration before test.')
    parser.add_argument('--calib_iters', default=None, type=int, help='the iterations for bn calibration')


    # data-related
    parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--dataset', default='st2stv2',
                        choices=list(DATASET_CONFIG.keys()) + ['cifar10', 'imagenet', 'cifar100'], help='path to dataset file list')
    parser.add_argument('--threed_data', action='store_true',
                        help='load data in the layout for 3D conv')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, directly crop the input_size')
    parser.add_argument('--random_sampling', action='store_true',
                        help='perform determinstic sampling for data loader')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='[v1] TSN data argmentation, [v2] resize the shorter side to `scale_range`')
    parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])
    parser.add_argument('--mean', type=float, nargs="+",
                        metavar='MEAN', help='mean, dimension should be 3 for RGB, 1 for flow')
    parser.add_argument('--std', type=float, nargs="+",
                        metavar='STD', help='std, dimension should be 3 for RGB, 1 for flow')
    parser.add_argument('--skip_normalization', action='store_true',
                        help='skip mean and std normalization, default use imagenet`s mean and std.')
    parser.add_argument('--use_lmdb', action='store_true',
                        help='use lmdb instead of jpeg.')
    parser.add_argument('--use_pyav', action='store_true', help='directly decode the videos.')
    parser.add_argument('--use_sparsity', action='store_true',
                        help='use sparsity')
    parser.add_argument('--sparsity_weight', default=1.0, type=float, metavar='SPARSITY_W',
                        help='sparsity_weight')

    # logging
    parser.add_argument('--logdir', default='', type=str, help='log path')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='frequency to print the log during the training')
    parser.add_argument('--show_model', action='store_true', help='show model summary')

    # for testing and validation
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    parser.add_argument('--num_clips', default=1, type=int)

    parser.add_argument('--cpu', action='store_true', help='using cpu only')


    # for distributed learning
    parser.add_argument('--sync-bn', action='store_true',
                        help='sync BN across GPUs')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--hostfile', default='', type=str,
                        help='hostfile distributed learning')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--pred_weights', type=float, nargs="+",
                        help='scale range for augmentor v2')

    parser.add_argument('--pred_files', type=str, nargs="+",
                        help='scale range for augmentor v2')
    parser.add_argument('--label_file', type=str,
                        help='scale range for augmentor v2')
    parser.add_argument('--after_softmax', action='store_true', help="perform softmax before ensumble")

    return parser
