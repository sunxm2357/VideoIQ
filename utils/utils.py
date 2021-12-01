import sys
sys.path.insert(0, '../')
import numpy as np
import shutil
import os
import subprocess
import time
import multiprocessing

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)
from models.twod_models.temporal_modeling import TAM
from utils.flops_benchmark import add_flops_counting_methods


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


class KDLoss(nn.modules.loss._Loss):
    """ inplace distillation for image classification.
    Refer to https://github.com/JiahuiYu/slimmable_networks/blob/master/utils/loss_ops.py
    """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        # import pdb
        # pdb.set_trace()
        # print(target, output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        return cross_entropy_loss


def getTAMloss(model):
    loss = 0
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, TAM):
            weights = torch.stack(
                [m._modules['prev_se'].fc1.weight.data.view(-1), m._modules['curr_se'].fc1.weight.data.view(-1),
                 m._modules['next_se'].fc1.weight.data.view(-1)], dim=0)
            L2 = torch.norm(weights, p=1, dim=0)
            loss += torch.sum(L2)
            cnt += L2.shape[0]
    return loss / cnt

def cal_acc_map(logits, test_y, topk=None):
    """

    :param logits: (NxK)
    :param test_y: (Nx1)
    :param topk (tuple(int)):
    :return:
        - list[float]: topk acc
        - float: mAP
    """
    return actnet_acc(logits, test_y, topk)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def actnet_acc(logits, test_y, topk=None, have_softmaxed=False):
    from torchnet import meter

    """

    :param logits: (NxK)
    :param test_y: (Nx1)
    :param topk (tuple(int)):
    :return:
        - list[float]: topk acc
        - float: mAP
    """
    num_classes = logits.shape[1]
    topk = [1, min(5, num_classes)] if topk is None else topk
    single_label = True if len(test_y.shape) == 1 else False
    probs = F.softmax(logits, dim=1) if not have_softmaxed else logits
    if single_label:
        acc_meter = meter.ClassErrorMeter(topk=topk, accuracy=True)
        acc_meter.add(logits, test_y)
        acc = acc_meter.value()
        gt = torch.zeros_like(logits)
        gt[torch.LongTensor(range(gt.size(0))), test_y.view(-1)] = 1
    else:
        gt = test_y
        acc = [0] * len(topk)
    map_meter = meter.mAPMeter()
    map_meter.add(probs, gt)
    ap = map_meter.value() * 100.0
    return acc, ap.item()

def map_charades(y_pred, y_true):
    """ Returns mAP """
    y_true = y_true.cpu().numpy().astype(np.int32)
    y_pred = y_pred.cpu().detach().numpy()
    m_aps = []
    n_classes = y_pred.shape[1]
    for oc_i in range(n_classes):
        pred_row = y_pred[:, oc_i]
        sorted_idxs = np.argsort(-pred_row)
        true_row = y_true[:, oc_i]
        tp = true_row[sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(np.nan)
            continue
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs + t_pcs).astype(float)
        avg_prec = 0
        for i in range(y_pred.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    m_ap = m_ap if not np.isnan(m_ap) else 0.0
    return [m_ap], [0]


def save_checkpoint(state, is_best, filepath='', is_policy=False, prefix=None):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        if prefix is None:
            if is_policy:
                shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'policy_best.pth.tar'))
            else:
                shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
        else:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, '%s_best.pth.tar' % prefix))


def save_checkpoint_eval(state, filepath=''):
    torch.save(state, os.path.join(filepath, 'model_best_bn_calib.pth.tar'))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_total_flops_params(summary):
    for line in summary.split("\n"):
        line = line.strip()
        if line == "":
            continue
        if "Total flops" in line:
            total_flops = line.split(":")[-1].strip()
        elif "Total params" in line:
            total_params = line.split(":")[-1].strip()

    return total_flops, total_params


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def bn_cali_fix(model):
    if isinstance(model, nn.DataParallel):
        for name, param in model.module.named_parameters():
            if 'bn' in name and 'num_batches_tracked' in name:
                param.data = 0

    else:
        for name, param in model.named_parameters():
            if 'bn' in name and 'num_batches_tracked' in name:
                param.data = 0
    return model


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader

def compute_flops(model, input, only_lstm, kwargs_dict):
    model = add_flops_counting_methods(model)
    model = model.train()

    model.start_flops_count()

    if only_lstm:
        _ = model.lstm_forward(input, **kwargs_dict)
    else:
        _ = model(input, **kwargs_dict)
    gflops = model.compute_average_flops_cost()

    return gflops


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    result = torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
    return result


def get_TSM_regularization(model, blending_frames):
    sparsity_loss = 0
    for m in model.modules():
        if isinstance(m, LearnableTSM):
            if blending_frames == 3:
                weights = torch.stack(
                        [m._modules['prev_se'].fc1.weight.view(-1), m._modules['curr_se'].fc1.weight.view(-1),
                        m._modules['next_se'].fc1.weight.view(-1)], dim=0)
            else:
                weights = torch.stack(
                    [m._modules['blending_layers']._modules['{}'.format(i)].fc1.weight.view(-1) for i in
                    range(blending_frames)], dim=0)
            l1_term = 1.0 - torch.sum(weights, dim=0)
            l2_term = torch.sum(weights ** 2, dim=0)
            sparsity_loss += (torch.sum(l1_term * l1_term) + torch.sum(torch.ones_like(l2_term) - l2_term)) / (l1_term.numel() * 2)
            #                print (loss, sparsity_loss)
    return sparsity_loss


def train(data_loader, model, criterion, optimizer, epoch, args, display=100,
          steps_per_epoch=99999999999, label_smoothing=0.0, num_classes=None,
          clip_gradient=None, gpu_id=None, rank=0, eval_criterion=accuracy, use_sparsity=False, sparsity_w=1.0,
          kwargs={}):
    if 'KD' in args.loss_type:
        kd_criterion = KDLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    tam_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            output = model(images, **kwargs)

            #TODO check label_smoothing
            if label_smoothing > 0.0:
                smoothed_target = torch.zeros([images.size(0), num_classes]).scatter_(
                    1, target.unsqueeze(1), 1.0) * (1.0 - label_smoothing) + 1 / float(num_classes) * label_smoothing
                smoothed_target = smoothed_target.type(torch.float)
                smoothed_target = smoothed_target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                loss = cross_entropy(output, smoothed_target)
            else:
                target = target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                if args.loss_type == 'CE':
                    loss = criterion(output, target)
                else:
                    raise ValueError('Loss Type (%s) is not supported' % args.loss_type)

            if use_sparsity:
                TAM_loss = getTAMloss(model)
                loss += TAM_loss * sparsity_w

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)

                clip_val = 'ClipVal: [{:03d}/{:03d}]'.format(epoch + 1, args.epochs)
                for name, param in model.named_parameters():
                    if 'q_alpha' in name:
                        clip_val += '\t {:1.2f} '.format(param.data.detach().cpu().numpy()[0])
                print(clip_val, flush=True)

            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def get_efficient_loss(policy, codebook):
    shape = policy.shape
    policy = policy.view(-1, shape[-1])
    assert shape[-1] == codebook.shape[0]
    codebook = codebook.unsqueeze(0).expand_as(policy)
    device = policy.get_device()
    if device != -1:
        codebook = codebook.to(device)
    loss = (policy * codebook).mean(0).sum()
    return loss


def get_entropy(prob):
    shape = prob.shape
    prob = prob.view(-1, shape[-1])
    entropy = (- torch.exp(prob) * prob).sum(dim=-1)
    return entropy


def train_ada(data_loader, model, criterion, optimizers, epoch, args, display=100,
              steps_per_epoch=99999999999, label_smoothing=0.0, num_classes=None,
              clip_gradient=None, gpu_id=None, rank=0, eval_criterion=accuracy, use_sparsity=False, sparsity_w=1.0,
              deterministic=False, kwargs={}):
    if 'KD' in args.loss_type:
        kd_criterion = KDLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    kd_losses = AverageMeter()
    eff_losses = AverageMeter()
    entropy_losses = AverageMeter()
    bal_losses = AverageMeter()
    pce_losses = AverageMeter()
    tam_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    policy_epoch = AverageMeter()

    if args.efficient:
        bit_width_family = [min(bit, 8) for bit in args.bit_width_family]
        if args.use_fp_as_bb:
            bit_width_family = [8] + bit_width_family
        codebook1 = ((torch.tensor(bit_width_family)) ** 2).float() / 64.
        # import pdb
        # pdb.set_trace()
        if len(args.skip_list) != 0:
            codebook2 = - (torch.tensor(args.skip_list).float() - 1) * 0 / 64.
            # codebook2 = - ((torch.tensor(args.skip_list) - 1) * max(bit_width_family) ** 2).float() / 64.
            codebook = torch.cat([codebook1, codebook2]).float()
        else:
            codebook = codebook1.float()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            output, policy, prob, output_h, output_32fp, policy_out = model(images, is_ada=True, deterministic=deterministic, **kwargs)
            # output = model(images, **kwargs)
            #TODO check label_smoothing
            if label_smoothing > 0.0:
                smoothed_target = torch.zeros([images.size(0), num_classes]).scatter_(
                    1, target.unsqueeze(1), 1.0) * (1.0 - label_smoothing) + 1 / float(num_classes) * label_smoothing
                smoothed_target = smoothed_target.type(torch.float)
                smoothed_target = smoothed_target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                loss = cross_entropy(output, smoothed_target)
            else:
                target = target.cuda(gpu_id, non_blocking=True)
                # target = target.cuda(non_blocking=True)
                if args.loss_type == 'CE':
                    loss = criterion(output, target)
                    ce_losses.update(loss)
                elif args.loss_type == 'KD':
                    target_soft = torch.nn.functional.softmax(output_h.detach().clone(), dim=1)
                    loss = kd_criterion.forward(output, target_soft)
                    kd_losses.update(loss)
                elif args.loss_type == 'KD_CE':
                    loss = criterion(output, target)
                    ce_losses.update(loss.clone())
                    if args.is_32fp:
                        target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                    else:
                        target_soft = torch.nn.functional.softmax(output_h.detach().clone(), dim=1)

                    kd_loss = kd_criterion.forward(output, target_soft)
                    loss += kd_loss
                    kd_losses.update(kd_loss)
                else:
                    raise ValueError('Loss Type (%s) is not supported' % args.loss_type)

            if args.is_policy_pred:
                policy_loss = criterion(policy_out, target)
                loss += args.policy_w * policy_loss
                pce_losses.update(policy_loss)

            if args.balanced:
                balanced_loss = torch.pow((policy.mean(dim=0).mean(dim=0) - 1.0 / (len(args.bit_width_family) +
                                                                        len(args.skip_list))), 2).sum()
                # if i == 0:
                #     print('Efficient Loss: ', efficient_loss)
                bal_losses.update(balanced_loss)
                loss += args.balanced_w * balanced_loss
                if args.debug:
                    print('Balanced Loss: ', balanced_loss)

            if use_sparsity:
                TAM_loss = getTAMloss(model)
                loss += TAM_loss * sparsity_w

            if args.efficient:
                efficient_loss = get_efficient_loss(policy, codebook)
                eff_losses.update(efficient_loss)
                loss += efficient_loss * args.efficient_w
                if args.debug:
                    print('Efficient Loss: ', efficient_loss)

            if args.entropy:
                entropy = get_entropy(prob)
                mean_entropy = entropy.mean()
                entropy_losses.update(mean_entropy)
                loss += mean_entropy * args.entropy_w
                if args.debug:
                    print('Deterministic Loss: ', mean_entropy)

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), images.size(0))
            policy_epoch.update(policy.detach().data.cpu().mean(0).mean(dim=0).numpy())
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'CE Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'KD Loss {kd_loss.val:.4f} ({kd_loss.avg:.4f})\t'
                      'Policy CE Loss {pce_loss.val:.4f} ({pce_loss.avg:.4f})\t'
                      'Efficient Loss {eff_loss.val:.4f} ({eff_loss.avg:.4f})\t'
                      'Entropy Loss {entropy_loss.val:.4f} ({entropy_loss.avg:.4f})\t'
                      'Balanced Loss {bal_loss.val:.4f} ({bal_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, ce_loss=ce_losses, kd_loss=kd_losses,
                       pce_loss=pce_losses, eff_loss=eff_losses, entropy_loss=entropy_losses,
                       bal_loss=bal_losses,  top1=top1, top5=top5), flush=True)
                print('Policy (current epoch): ', policy_epoch.avg.tolist(), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, policy_epoch.avg, num_batch, ce_losses.avg, \
           kd_losses.avg, pce_losses.avg, eff_losses.avg, entropy_losses.avg, bal_losses.avg


def train_ap(data_loader, model, criterion, optimizer, epoch, w_bit_width_family, a_bit_width_family, loss_type, args,  display=100,
             steps_per_epoch=99999999999, label_smoothing=0.0, num_classes=None,
             clip_gradient=None, gpu_id=None, rank=0, eval_criterion=accuracy, use_sparsity=False, sparsity_w=1.0,
             kwargs={}):

    if 'KD' in loss_type:
        kd_criterion = KDLoss()

    assert len(w_bit_width_family) == len(a_bit_width_family)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses, top1s, top5s = [], [], []
    for _ in w_bit_width_family:
        losses.append(AverageMeter())
        top1s.append(AverageMeter())
        top5s.append(AverageMeter())

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    if args.debug and args.rank == 0:
        print(args.use_fp)

    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            for idx, (w_bit, a_bit) in enumerate(zip(w_bit_width_family, a_bit_width_family)):
                output, output_32fp = model(images, is_ada=False, w_bit=w_bit, a_bit=a_bit)
                #TODO check label_smoothing
                # print('Computing Loss')
                if label_smoothing > 0.0:
                    smoothed_target = torch.zeros([images.size(0), num_classes]).scatter_(
                        1, target.unsqueeze(1), 1.0) * (1.0 - label_smoothing) + 1 / float(num_classes) * label_smoothing
                    smoothed_target = smoothed_target.type(torch.float)
                    smoothed_target = smoothed_target.cuda(gpu_id, non_blocking=True)
                    loss = cross_entropy(output, smoothed_target)
                else:
                    target = target.cuda(gpu_id, non_blocking=True)
                    if loss_type == 'CE':
                        loss = criterion(output, target)
                    elif loss_type == 'KD':
                        if idx == 0:
                            loss = criterion(output, target)
                            if args.is_32fp:
                                target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                                loss += args.kd_weight * kd_criterion.forward(output, target_soft)
                            else:
                                target_soft = torch.nn.functional.softmax(output.detach().clone(), dim=1)
                        else:
                            loss = kd_criterion.forward(output, target_soft)
                    elif loss_type == 'KD_CE':
                        if idx == 0:
                            loss = criterion(output, target)
                            if args.is_32fp:
                                target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                                loss += args.kd_weight * kd_criterion.forward(output, target_soft)
                            else:
                                target_soft = torch.nn.functional.softmax(output.detach().clone(), dim=1)
                        else:
                            loss = criterion(output, target)
                            loss += args.kd_weight * kd_criterion.forward(output, target_soft)
                    else:
                        raise ValueError('Loss Type (%s) is not supported' % loss_type)

                if use_sparsity:
                    TAM_loss = getTAMloss(model)
                    loss += TAM_loss * sparsity_w

                # measure accuracy and record loss
                prec1, prec5 = eval_criterion(output, target)

                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    dist.all_reduce(prec1)
                    dist.all_reduce(prec5)
                    prec1 /= world_size
                    prec5 /= world_size

                # import pdb
                # pdb.set_trace()
                losses[idx].update(loss.item(), images.size(0))
                top1s[idx].update(prec1[0], images.size(0))
                top5s[idx].update(prec5[0], images.size(0))
                # if args.rank == 0:
                #     print('DEBUG: backward, w_bit %d' % w_bit, flush=True)
                loss.backward(retain_graph=True)

                # total_loss += loss
            # compute gradient and do SGD step
            # total_loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)


            # if args.rank == 0:
            #     print('DEBUG: step', flush=True)
            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                for idx in range(len(w_bit_width_family)):
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           epoch, i, len(data_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses[idx], top1=top1s[idx], top5=top5s[idx]), flush=True)

                clip_val = 'ClipVal: [{:03d}/{:03d}]'.format(epoch + 1, args.epochs)
                for name, param in model.named_parameters():
                    if 'q_alpha' in name:
                        clip_val += '\t {:1.2f} '.format(param.data.detach().cpu().numpy()[0])
                print(clip_val, flush=True)

            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    top1s_avg = []
    top5s_avg = []
    losses_avg = []
    for top1, top5, loss_ in zip(top1s, top5s, losses):
        top1s_avg.append(top1.avg)
        top5s_avg.append(top5.avg)
        losses_avg.append(loss_.avg)

    return top1s_avg, top5s_avg, losses_avg, batch_time.avg, data_time.avg, num_batch


def train_ap_mm(data_loader, model, criterion, optimizer, epoch, w_bit_width_family, a_bit_width_family, loss_type, args,  display=100,
             steps_per_epoch=99999999999, label_smoothing=0.0, num_classes=None,
             clip_gradient=None, gpu_id=None, rank=0, eval_criterion=accuracy, use_sparsity=False, sparsity_w=1.0,
             kwargs={}):

    if 'KD' in loss_type:
        kd_criterion = KDLoss()

    assert len(w_bit_width_family) == len(a_bit_width_family)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses, top1s, top5s = [], [], []
    for _ in w_bit_width_family:
        losses.append(AverageMeter())
        top1s.append(AverageMeter())
        top5s.append(AverageMeter())

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    end = time.time()
    num_batch = 0
    if args.debug and args.rank == 0:
        print(args.use_fp)

    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            outputs, output_32fp = model(images)
            total_loss = 0
            for idx, (w_bit, a_bit) in enumerate(zip(w_bit_width_family, a_bit_width_family)):
                # output = model(images, w_bit, a_bit)
                #TODO check label_smoothing
                # print('Computing Loss')
                if label_smoothing > 0.0:
                    smoothed_target = torch.zeros([images.size(0), num_classes]).scatter_(
                        1, target.unsqueeze(1), 1.0) * (1.0 - label_smoothing) + 1 / float(num_classes) * label_smoothing
                    smoothed_target = smoothed_target.type(torch.float)
                    smoothed_target = smoothed_target.cuda(gpu_id, non_blocking=True)
                    loss = cross_entropy(outputs[idx], smoothed_target)
                else:
                    target = target.cuda(gpu_id, non_blocking=True)
                    if loss_type == 'CE':
                        loss = criterion(outputs[idx], target)
                    elif loss_type == 'KD':
                        if idx == 0:
                            loss = criterion(outputs[idx], target)
                            if args.is_32fp:
                                target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                                loss += args.kd_weight * kd_criterion.forward(outputs[idx], target_soft)
                            else:
                                target_soft = torch.nn.functional.softmax(outputs[idx].detach().clone(), dim=1)
                        else:
                            loss = kd_criterion.forward(outputs[idx], target_soft)
                    elif loss_type == 'KD_CE':
                        if idx == 0:
                            loss = criterion(outputs[idx], target)
                            if args.is_32fp:
                                target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                                loss += args.kd_weight * kd_criterion.forward(outputs[idx], target_soft)
                            else:
                                target_soft = torch.nn.functional.softmax(outputs[idx].detach().clone(), dim=1)
                        else:
                            loss = criterion(outputs[idx], target)
                            loss += args.kd_weight * kd_criterion.forward(outputs[idx], target_soft)
                    else:
                        raise ValueError('Loss Type (%s) is not supported' % loss_type)

                if use_sparsity:
                    TAM_loss = getTAMloss(model)
                    loss += TAM_loss * sparsity_w

                # measure accuracy and record loss
                prec1, prec5 = eval_criterion(outputs[idx], target)

                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    dist.all_reduce(prec1)
                    dist.all_reduce(prec5)
                    prec1 /= world_size
                    prec5 /= world_size

                # import pdb
                # pdb.set_trace()
                losses[idx].update(loss.item(), images.size(0))
                top1s[idx].update(prec1[0], images.size(0))
                top5s[idx].update(prec5[0], images.size(0))
                # if args.rank == 0:
                #     print('DEBUG: backward, w_bit %d' % w_bit, flush=True)
                # loss.backward(retain_graph=True)
                total_loss += loss
                # total_loss += loss
            # compute gradient and do SGD step
            # total_loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)


            # if args.rank == 0:
            #     print('DEBUG: step', flush=True)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                for idx in range(len(w_bit_width_family)):
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           epoch, i, len(data_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses[idx], top1=top1s[idx], top5=top5s[idx]), flush=True)

                clip_val = 'ClipVal: [{:03d}/{:03d}]'.format(epoch + 1, args.epochs)
                for name, param in model.named_parameters():
                    if 'q_alpha' in name:
                        clip_val += '\t {:1.2f} '.format(param.data.detach().cpu().numpy()[0])
                print(clip_val, flush=True)

            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    top1s_avg = []
    top5s_avg = []
    losses_avg = []
    for top1, top5, loss_ in zip(top1s, top5s, losses):
        top1s_avg.append(top1.avg)
        top5s_avg.append(top5.avg)
        losses_avg.append(loss_.avg)

    return top1s_avg, top5s_avg, losses_avg, batch_time.avg, data_time.avg, num_batch


def bn_calibration(data_loader, model, gpu_id=None, calib_iters=None):
    # switch to train mode
    model.train()
    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            model(images)
            t_bar.update(1)
            if calib_iters is not None:
                if i >= calib_iters:
                    break


def bn_calibration_ap(data_loader, model, w_bit_width_family, a_bit_width_family,  gpu_id=None,  calib_iters=None):
    # switch to train mode
    model.train()
    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            for idx, (w_bit, a_bit) in enumerate(zip(w_bit_width_family, a_bit_width_family)):
                model(images, w_bit, a_bit)
            t_bar.update(1)
            if calib_iters is not None:
                if i >= calib_iters:
                    break


def validate(data_loader, model, criterion, args, gpu_id=None, eval_criterion=accuracy):
    if 'KD' in args.loss_type:
        kd_criterion = KDLoss()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output = model(images)

            if args.loss_type == 'CE':
                loss = criterion(output, target)
            else:
                raise ValueError('Loss Type (%s) is not supported' % args.loss_type)

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)
    torch.cuda.empty_cache()
    return top1.avg, top5.avg, losses.avg, batch_time.avg


def validate_ada(data_loader, model, criterion, args, gpu_id=None, eval_criterion=accuracy):
    if 'KD' in args.loss_type:
        kd_criterion = KDLoss()
    batch_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    kd_losses = AverageMeter()
    eff_losses = AverageMeter()
    entropy_losses = AverageMeter()
    bal_losses = AverageMeter()
    pce_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    flops = AverageMeter()
    policy_epoch = AverageMeter()


    bit_width_family = [min(bit, 8) for bit in args.bit_width_family]
    if args.use_fp_as_bb:
        bit_width_family = [8] + bit_width_family
    codebook1 = ((torch.tensor(bit_width_family)) ** 2).float() / 64.
    if len(args.skip_list) != 0:
        codebook2 = - (torch.tensor(args.skip_list).float() - 1) * 0 / 64.
        codebook = torch.cat([codebook1, codebook2]).float()
    else:
        codebook = codebook1.float()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
            target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            output, policy, prob, output_h, output_32fp, policy_out = model(images, is_ada=True)
            if args.loss_type == 'CE':
                loss = criterion(output, target)
                ce_losses.update(loss)
            elif args.loss_type == 'KD':
                target_soft = torch.nn.functional.softmax(output_h.detach().clone(), dim=1)
                loss = kd_criterion.forward(output, target_soft)
                kd_losses.update(loss)
            elif args.loss_type == 'KD_CE':
                loss = criterion(output, target)
                ce_losses.update(loss)
                if args.is_32fp:
                    target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                else:
                    target_soft = torch.nn.functional.softmax(output_h.detach().clone(), dim=1)
                kd_loss = kd_criterion.forward(output, target_soft)
                loss += kd_loss
                kd_losses.update(kd_loss)
            else:
                raise ValueError('Loss Type (%s) is not supported' % args.loss_type)

            if args.is_policy_pred:
                policy_loss = criterion(policy_out, target)
                loss += args.policy_w * policy_loss
                pce_losses.update(policy_loss)

            if args.balanced:
                balanced_loss = torch.pow((policy.mean(dim=0).mean(dim=0) - 1.0 / (len(args.bit_width_family) +
                                                                        len(args.skip_list))), 2).sum()
                # if i == 0:
                #     print('Efficient Loss: ', efficient_loss)
                bal_losses.update(balanced_loss)
                loss += args.balanced_w * balanced_loss

            if args.entropy:
                entropy = get_entropy(prob)
                mean_entropy = entropy.mean()
                entropy_losses.update(mean_entropy)
                loss += mean_entropy * args.entropy_w
                if args.debug:
                    print('Deterministic Loss: ', mean_entropy)

            efficient_loss = get_efficient_loss(policy, codebook)
            if 'resnet' in args.backbone_net and args.depth == 18:
                efficient_loss = efficient_loss * (29.1 - 1.92) + 1.92
            elif 'resnet' in args.backbone_net and args.depth == 50:
                efficient_loss = efficient_loss * (65.76 - 1.92) + 1.92
            else:
                raise ValueError
            flops.update(efficient_loss)

            if args.efficient:
                eff_losses.update(efficient_loss)

            # measure accuracy and record loss
            prec1, prec5 = eval_criterion(output, target)
            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            policy_epoch.update(policy.detach().data.cpu().mean(0).mean(dim=0).numpy())
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)
    torch.cuda.empty_cache()
    return top1.avg, top5.avg, losses.avg, batch_time.avg, policy_epoch.avg, flops.avg, ce_losses.avg, \
           kd_losses.avg, pce_losses.avg, eff_losses.avg, entropy_losses.avg, bal_losses.avg


def validate_ap(data_loader, model, criterion,  w_bit_width_family, a_bit_width_family, loss_type, args,
                gpu_id=None, eval_criterion=accuracy):
    if 'KD' in loss_type:
        kd_criterion = KDLoss()

    batch_time = AverageMeter()
    assert len(w_bit_width_family) == len(a_bit_width_family)
    losses, top1s, top5s = [], [], []
    for _ in range(len(w_bit_width_family)):
        losses.append(AverageMeter())
        top1s.append(AverageMeter())
        top5s.append(AverageMeter())

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
                target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            for idx, (w_bit, a_bit) in enumerate(zip(w_bit_width_family, a_bit_width_family)):
                output, output_32fp = model(images, is_ada=False, w_bit=w_bit, a_bit=a_bit)
                # output = model(images)
                if loss_type == 'CE':
                    loss = criterion(output, target)
                elif loss_type == 'KD':
                    if idx == 0:
                        loss = criterion(output, target)
                        if args.is_32fp:
                            target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                        else:
                            target_soft = torch.nn.functional.softmax(output.detach().clone(), dim=1)
                    else:
                        loss = kd_criterion.forward(output, target_soft)
                elif loss_type == 'KD_CE':
                    if idx == 0:
                        loss = criterion(output, target)
                        if args.is_32fp:
                            target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                        else:
                            target_soft = torch.nn.functional.softmax(output.detach().clone(), dim=1)
                    else:
                        loss = criterion(output, target)
                        loss += args.kd_weight * kd_criterion.forward(output, target_soft)
                else:
                    raise ValueError('Loss Type (%s) is not supported' % loss_type)

                # measure accuracy and record loss
                prec1, prec5 = eval_criterion(output, target)
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    dist.all_reduce(prec1)
                    dist.all_reduce(prec5)
                    prec1 /= world_size
                    prec5 /= world_size
                losses[idx].update(loss.item(), images.size(0))
                top1s[idx].update(prec1[0], images.size(0))
                top5s[idx].update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)
    torch.cuda.empty_cache()
    top1s_avg = []
    top5s_avg = []
    losses_avg = []
    for top1, top5, loss_ in zip(top1s, top5s, losses):
        top1s_avg.append(top1.avg)
        top5s_avg.append(top5.avg)
        losses_avg.append(loss_.avg)
    return top1s_avg, top5s_avg, losses_avg, batch_time.avg


def validate_ap_mm(data_loader, model, criterion,  w_bit_width_family, a_bit_width_family, loss_type, args,
                 gpu_id=None, eval_criterion=accuracy):
    if 'KD' in loss_type:
        kd_criterion = KDLoss()

    batch_time = AverageMeter()
    assert len(w_bit_width_family) == len(a_bit_width_family)
    losses, top1s, top5s = [], [], []
    for _ in range(len(w_bit_width_family)):
        losses.append(AverageMeter())
        top1s.append(AverageMeter())
        top5s.append(AverageMeter())

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)
                target = target.cuda(gpu_id, non_blocking=True)

            # compute output
            outputs, output_32fp = model(images)

            for idx, (w_bit, a_bit) in enumerate(zip(w_bit_width_family, a_bit_width_family)):
                # output = model(images)
                if loss_type == 'CE':
                    loss = criterion(outputs[idx], target)
                elif loss_type == 'KD':
                    if idx == 0:
                        loss = criterion(outputs[idx], target)
                        if args.is_32fp:
                            target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                        else:
                            target_soft = torch.nn.functional.softmax(outputs[idx].detach().clone(), dim=1)
                    else:
                        loss = kd_criterion.forward(outputs[idx], target_soft)
                elif loss_type == 'KD_CE':
                    if idx == 0:
                        loss = criterion(outputs[idx], target)
                        if args.is_32fp:
                            target_soft = torch.nn.functional.softmax(output_32fp.detach().clone(), dim=1)
                        else:
                            target_soft = torch.nn.functional.softmax(outputs[idx].detach().clone(), dim=1)
                    else:
                        loss = criterion(outputs[idx], target)
                        loss += args.kd_weight * kd_criterion.forward(outputs[idx], target_soft)
                else:
                    raise ValueError('Loss Type (%s) is not supported' % loss_type)

                # measure accuracy and record loss
                prec1, prec5 = eval_criterion(outputs[idx], target)
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    dist.all_reduce(prec1)
                    dist.all_reduce(prec5)
                    prec1 /= world_size
                    prec5 /= world_size
                losses[idx].update(loss.item(), images.size(0))
                top1s[idx].update(prec1[0], images.size(0))
                top5s[idx].update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)
    torch.cuda.empty_cache()
    top1s_avg = []
    top5s_avg = []
    losses_avg = []
    for top1, top5, loss_ in zip(top1s, top5s, losses):
        top1s_avg.append(top1.avg)
        top5s_avg.append(top5.avg)
        losses_avg.append(loss_.avg)
    return top1s_avg, top5s_avg, losses_avg, batch_time.avg



if __name__ == '__main__':
    avg = AverageMeter()
    tmp1 = torch.rand(10)
    tmp2 = torch.rand(10)
    avg.update(tmp1)
    avg.update(tmp2)
    import pdb
    pdb.set_trace()
