import os
import time

import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from tqdm import tqdm

from models import build_model
from utils.utils import build_dataflow, AverageMeter, \
    actnet_acc, get_augmentor, bn_cali_fix, bn_calibration, get_efficient_loss, get_entropy
from utils.video_transforms import *
from utils.video_dataset import VideoDataSet, VideoDataSetLMDB
from utils.video_dataset2 import MultiVideoDataSetOnline
from utils.dataset_config import get_dataset_config
from opts import arg_parser


def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        cls_id = 0
        for label in f.readlines():
            label = label.strip()
            if label == "":
                continue
            id_to_label[cls_id] = label
            label_to_id[label] = cls_id
            cls_id += 1
    return id_to_label, label_to_id


def eval_a_batch(data, model, in_channels, num_clips=1, num_crops=1, rand_policy=False, dist=None, deterministic=False,
                 modality='rgb', softmax=False, threed_data=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result, policy, prob, _, _, _ = model(data, is_ada=True, rand_policy=rand_policy, dist=dist, deterministic=deterministic)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result, policy, prob


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file, multilabel = get_dataset_config(args.dataset, args.use_lmdb)

    if args.dataset == 'activitynet':
        if args.server == 'diva':
            datadir = '/store/workspaces/rpanda/sunxm/datasets/activitynet'
        elif args.server in ['aimos', 'satori']:
            datadir = args.datadir
        else:
            raise ValueError('server %s is not supported' % args.server)
    else:
        datadir = args.datadir

    data_list_name = val_list_name if args.evaluate else test_list_name

    args.num_classes = num_classes
    if args.dataset == 'st2stv1' or args.dataset == 'activitynet':
        id_to_label, label_to_id = load_categories(os.path.join(datadir, label_file))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std

    model = model.cuda()
    model.eval()

    if args.threed_data:
        dummy_data = (args.input_channels, args.groups, args.input_size, args.input_size)
    else:
        dummy_data = (args.input_channels * args.groups, args.input_size, args.input_size)

#    model_summary = torchsummary.summary(model, input_size=dummy_data)

    # flops, params = extract_total_flops_params(model_summary)
    # flops = int(flops.replace(',', '')) * (args.num_clips * args.num_crops)
    model = torch.nn.DataParallel(model).cuda()
    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print('the epoch is %d' % checkpoint['epoch'])
    else:
        print("=> creating model '{}'".format(arch_name))

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    # augments = []
    # if args.num_crops == 1:
    #     augments += [
    #         GroupScale(scale_size),
    #         GroupCenterCrop(args.input_size)
    #     ]
    # else:
    #     flip = True if args.num_crops == 10 else False
    #     augments += [
    #         GroupOverSample(args.input_size, scale_size, num_crops=args.num_crops, flip=flip),
    #     ]
    # augments += [
    #     Stack(threed_data=args.threed_data),
    #     ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
    #     GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    # ]
    #
    # augmentor = transforms.Compose(augments)
    augmentor = get_augmentor(False, args.input_size, mean, std, args.disable_scaleup,
                                  threed_data=args.threed_data, version=args.augmentor_ver,
                                  scale_range=args.scale_range)

    # Data loading code
    data_list = os.path.join(datadir, data_list_name)
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_size))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}, offset from center with {}".format(args.num_clips, sample_offsets))

    if args.use_pyav:
        video_data_cls = MultiVideoDataSetOnline
    else:
        video_data_cls = VideoDataSetLMDB if args.use_lmdb else VideoDataSet

    val_dataset = video_data_cls(args.datadir, data_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips, modality=args.modality,
                                 image_tmpl=image_tmpl, dense_sampling=args.dense_sampling,
                                 fixed_offset=not args.random_sampling,
                                 transform=augmentor, is_train=False, test_mode=not args.evaluate,
                                 seperator=filename_seperator, filter_video=filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    if args.bn_calibrate:
        print('BN Calibration')
        train_list = os.path.join(datadir, train_list_name)

        train_augmentor = get_augmentor(True, args.input_size, mean, std, threed_data=args.threed_data,
                                        version=args.augmentor_ver, scale_range=args.scale_range)
        train_dataset = video_data_cls(args.datadir, train_list, args.groups, args.frames_per_group,
                                       num_clips=args.num_clips,
                                       modality=args.modality, image_tmpl=image_tmpl,
                                       dense_sampling=args.dense_sampling,
                                       transform=train_augmentor, is_train=True, test_mode=False,
                                       seperator=filename_seperator, filter_video=filter_video)

        train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                      workers=args.workers)
        model = bn_cali_fix(model)
        bn_calibration(train_loader, model, args.gpu)


    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    batch_time = AverageMeter()
    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
    else:
        logfile = open(os.path.join(log_folder,
                                    'test_{}crops_{}clips_{}.csv'.format(args.num_crops,
                                                                         args.num_clips,
                                                                         args.input_size))
                       , 'w')

    total_outputs = 0
    outputs = torch.zeros((len(data_loader) * args.batch_size, num_classes))
    num_frames = args.groups * args.frames_per_group
    decision_space = len(args.skip_list) + len(args.bit_width_family)
    decision_space = decision_space + 1 if args.use_fp_as_bb else decision_space
    policys = torch.zeros((len(data_loader) * args.batch_size * num_frames, decision_space))
    probs = torch.zeros((len(data_loader) * args.batch_size * num_frames, decision_space))
    if args.evaluate:
        if multilabel:
            labels = torch.zeros((len(data_loader) * args.batch_size, num_classes), dtype=torch.long)
        else:
            labels = torch.zeros((len(data_loader) * args.batch_size), dtype=torch.long)
    else:
        labels = [None] * len(data_loader) * args.batch_size
    # switch to evaluate mode
    model.eval()
    total_batches = len(data_loader)
    print('Temperature of Policy Net: ', model.module.policy_net.temperature)
    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        for i, (video, label) in enumerate(data_loader):
            output, policy, prob = eval_a_batch(video, model, args.input_channels, num_clips=args.num_clips, deterministic=args.deterministic,
                                  num_crops=args.num_crops, rand_policy=args.rand_policy, dist=args.dist,  modality=args.modality, softmax=True, threed_data=args.threed_data)
            batch_size = output.shape[0]
            outputs[total_outputs:total_outputs + batch_size, :] = output
            policys[total_outputs * num_frames:(total_outputs + batch_size) * num_frames] = policy.view(-1, decision_space)
            probs[total_outputs * num_frames:(total_outputs + batch_size) * num_frames] = prob.view(-1, decision_space)
            if multilabel:
                labels[total_outputs:total_outputs + batch_size, :] = label
            else:
                labels[total_outputs:total_outputs + batch_size] = label

            total_outputs += video.shape[0]
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

        outputs = outputs[:total_outputs]
        labels = labels[:total_outputs]
        print("Predicted {} videos.".format(total_outputs), flush=True)
        if args.rand_policy:
            npy_prefix = 'rand'
        else:
            npy_prefix = os.path.basename(args.pretrained).split(".")[0]
        np.save(os.path.join(log_folder, '{}_{}crops_{}clips_{}_details_{}.npy'.format("val" if args.evaluate else "test", args.num_crops, args.num_clips, args.input_size, npy_prefix)), outputs)
        np.save(os.path.join(log_folder, '{}_{}crops_{}clips.npy'.format('labels', args.num_crops, args.num_clips)), labels)


        if not args.evaluate:
            json_output = {
                'version': "VERSION 1.3",
                'results': {},
                'external_data': {'used': False, 'details': 'none'}
            }
            prob = F.softmax(outputs, dim=1).data.cpu().numpy().copy()
            predictions = np.argsort(prob, axis=1)
            for ii in range(len(predictions)):
                temp = predictions[ii][::-1][:5]
                preds = [str(pred) for pred in temp]
                if args.dataset == 'st2stv1':
                    print("{};{}".format(labels[ii], id_to_label[int(preds[0])]), file=logfile)
                elif args.dataset == 'activitynet':
                    video_id = labels[ii].replace("v_", "")
                    if video_id not in json_output['results']:
                        json_output['results'][video_id] = []
                    for jj in range(num_classes):
                        tmp = {'label': id_to_label[predictions[ii][::-1][jj]],
                               'score': prob[ii, predictions[ii][::-1][jj]].item()}
                        json_output['results'][video_id].append(tmp)
                else:
                    print("{};{}".format(labels[ii], ";".join(preds)), file=logfile)

            if args.dataset == 'activitynet':
                json.dump(json_output, logfile, indent=4)
        else:
            acc, mAP = actnet_acc(outputs, labels)
            top1, top5 = acc
            print(args.pretrained, file=logfile)
            print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\tmAP: {:.4f}\tFLOPs: {:,}\tParams:{} '.format(
                args.input_size, scale_size, args.num_crops, args.num_clips, top1, top5, mAP, 0, 0), flush=True)
            print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}\tmAP: {:.4f}\tFLOPs: {:,}\tParams:{} '.format(
                args.input_size, scale_size, args.num_crops, args.num_clips, top1, top5, mAP, 0, 0), flush=True, file=logfile)
            print('Policy Dist: ', policys.mean(0), flush=True)
            print('Policy Dist: ', policys.mean(0), flush=True, file=logfile)


            bit_width_family = [min(bit, 8) for bit in args.bit_width_family]
            if args.use_fp_as_bb:
                bit_width_family = [8] + bit_width_family
            codebook1 = ((torch.tensor(bit_width_family)) ** 2).float() / 64.
            if len(args.skip_list) != 0:
                codebook2 = - (torch.tensor(args.skip_list).float() - 1) * 0 / 64.
                codebook = torch.cat([codebook1, codebook2]).float()
            else:
                codebook = codebook1.float()
            efficient_loss = get_efficient_loss(policys, codebook)
            if 'resnet' in args.backbone_net and args.depth == 18:
                flops = efficient_loss * (29.1 - 1.92) + 1.92
            elif 'resnet' in args.backbone_net and args.depth == 50:
                flops = efficient_loss * (65.76 - 1.92) + 1.92
            else:
                raise ValueError

            policy_flops = 0.779 * min(args.p_bit, 8) ** 2 / 64. + 0.119
            total_flops = policy_flops + flops
            print('FLOPs (current epoch):\tBackbone: {:.2f}G,\tPolicyNet: {:.2f}G,\tTotal: {:.2f}G '.format(flops,
                                                                                                            policy_flops,
                                                                                                            total_flops),
                  flush=True)

            print('FLOPs (current epoch):\tBackbone: {:.2f}G,\tPolicyNet: {:.2f}G,\tTotal: {:.2f}G '.format(flops,
                                                                                                            policy_flops,
                                                                                                            total_flops),
                  flush=True, file=logfile)

            entropy = get_entropy(probs)
            print('Entropy of the Validation Set:  {:.2f}'.format(entropy.mean()), flush=True)
            print('Entropy of the Validation Set:  {:.2f}'.format(entropy.mean()), flush=True, file=logfile)

    logfile.close()


if __name__ == '__main__':
    main()
