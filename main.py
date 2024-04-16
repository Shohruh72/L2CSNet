import argparse
import os
import csv
import tqdm
import copy
import numpy as np
from timm import utils

import torch
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset


def train(args):
    model = nn.gaze_net(args.model_name, args.bins).cuda()
    ema = nn.EMA(model) if args.local_rank == 0 else None

    sampler = None
    dataset = Dataset(args.data_dir, is_train=True)
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler=sampler, num_workers=8, pin_memory=True)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    best = float('inf')
    num_steps = len(loader)
    softmax = torch.nn.Softmax(dim=1).cuda()
    reg_criterion = torch.nn.MSELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam([
        {'params': util.get_ignored_params(model), 'lr': 0},
        {'params': util.get_non_ignored_params(model), 'lr': args.lr},
        {'params': util.get_fc_params(model), 'lr': args.lr}
    ], args.lr)
    idx_tensor = torch.FloatTensor([idx for idx in range(90)]).cuda()
    scheduler = util.CosineLR(args, optimizer)
    with open('./weights/log.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'Pitch', 'Yaw'])
            logger.writeheader()
        for epoch in range(args.epochs):
            p_bar = loader
            avg_loss_pitch = util.AverageMeter()
            avg_loss_yaw = util.AverageMeter()

            if args.local_rank == 0:
                print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'p_loss', 'y_loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=num_steps)

            for images, labels in p_bar:
                images = images.cuda()

                p_gt = labels[0][:, 0].cuda()
                y_gt = labels[0][:, 1].cuda()

                p_cont_gt = labels[1][:, 0].cuda()
                y_cont_gt = labels[1][:, 1].cuda()

                pitch, yaw = model(images)

                # Cross entropy loss
                loss_p = criterion(pitch, p_gt)
                loss_y = criterion(yaw, y_gt)

                # MSE loss
                p_pred = torch.sum(softmax(pitch) * idx_tensor, 1) * 4 - 180
                y_pred = torch.sum(softmax(yaw) * idx_tensor, 1) * 4 - 180

                # Total loss
                loss_p += reg_criterion(p_pred, p_cont_gt)
                loss_y += reg_criterion(y_pred, y_cont_gt)

                loss_seq = [loss_p, loss_y]
                grad_seq = [torch.tensor(1.0).cuda() for _ in range(len(loss_seq))]

                if args.distributed:
                    loss_p = utils.reduce_tensor(loss_p, args.world_size)
                    loss_y = utils.reduce_tensor(loss_y, args.world_size)

                avg_loss_pitch.update(loss_p.item(), images.size(0))
                avg_loss_yaw.update(loss_y.item(), images.size(0))

                optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer.step()
                if ema:
                    ema.update(model)

                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.3g' + '%10.3g') % (
                        f'{epoch + 1}/{args.epochs}', memory, avg_loss_pitch.avg, avg_loss_yaw.avg)
                    p_bar.set_description(s)

            scheduler.step(epoch, optimizer)

            if args.local_rank == 0:
                last = test(args, ema.ema)
                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'Pitch': str(f'{last[0]:.3f}'),
                                 'Yaw': str(f'{last[1]:.3f}'),})
                log.flush()
                is_best = sum(last) < best

                if is_best:
                    best = sum(last)
                save = {'epoch': epoch, 'model': copy.deepcopy(ema.ema).half()}
                torch.save(save, f'weights/last.pt')

                if is_best:
                    torch.save(save, f'weights/best.pt')
                del save
    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')
        util.strip_optimizer('./weights/last.pt')
    torch.cuda.empty_cache()


def test(args, model=None):
    if model is None:
        model = torch.load('./weights/best.pt', 'cuda')
        model = model['model'].float()
    model.eval()

    dataset = Dataset(args.data_dir, is_train=False)
    loader = data.DataLoader(dataset, args.batch_size, False, num_workers=4, pin_memory=True)

    softmax = torch.nn.Softmax(dim=1)
    total, p_err, y_err = 0, .0, .0
    idx_tensor = torch.FloatTensor([idx for idx in range(90)]).cuda()

    for images, labels in tqdm.tqdm(loader, ('%10s' * 2) % ('Pitch', 'Yaw')):
        images = images.cuda()
        total += labels[1].size(0)
        p_gt = labels[1][:, 0].float() * np.pi / 180
        y_gt = labels[1][:, 1].float() * np.pi / 180
        pitch, yaw = model(images)

        p_pred, y_pred = softmax(pitch), softmax(yaw)

        p_pred = torch.sum(p_pred * idx_tensor, 1).cpu() * 4 - 180
        y_pred = torch.sum(y_pred * idx_tensor, 1).cpu() * 4 - 180

        pitch_pred = p_pred * np.pi / 180
        yaw_pred = y_pred * np.pi / 180

        for p, y, pl, yl in zip(pitch_pred, yaw_pred, p_gt, y_gt):
            pitch_error, yaw_error = util.calculate_errors([p, y], [pl, yl])
            p_err += pitch_error
            y_err += yaw_error

    p_error, y_error = p_err / total, y_err / total
    print(('%10s' * 2) % (f'{p_error:.3f}', f'{y_error:.3f}'))

    model.float()  # for training
    return p_error, y_error


# def demo(args):


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='18')
    parser.add_argument('--data-dir', type=str, default='./Gaze360')
    parser.add_argument('--bins', type=int, default=90)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', dest='lr', type=float, default=0.00001)
    parser.add_argument('--train',  action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
