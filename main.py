import argparse
import copy
import csv
import os

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from face_detection import RetinaFace
from timm import utils
from torch.utils import data

from nets import nn
from utils import util
from utils.datasets import Dataset


def train(args):
    model = nn.gaze_net(args.arch, args.bin)
    model.cuda()
    dataset = Dataset(args, True, True)
    loader = data.DataLoader(dataset, args.batch_size, True, num_workers=0, pin_memory=True)
    torch.backends.cudnn.benchmark = True

    best = float('inf')
    softmax = torch.nn.Softmax(dim=1).cuda()
    reg_criterion = torch.nn.MSELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer_gaze = torch.optim.Adam([
        {'params': util.get_ignored_params(model), 'lr': 0},
        {'params': util.get_non_ignored_params(model), 'lr': args.lr},
        {'params': util.get_fc_params(model), 'lr': args.lr}], args.lr)

    idx_tensor = torch.FloatTensor([idx for idx in range(args.bin)]).cuda()

    print(
        f"\n[TRAIN CONFIGS], batch={args.batch_size}, model_arch={args.arch}\nStart testing dataset={args.data_name}, loader={len(loader)}\n")
    with open('./weights/log.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch', 'MAE', 'Pitch', 'Yaw'])
            logger.writeheader()
        for epoch in range(args.epochs):
            p_bar = loader
            avg_loss_pitch = util.AverageMeter()
            avg_loss_yaw = util.AverageMeter()
            if args.local_rank == 0:
                print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'p_loss', 'y_loss'))
                p_bar = tqdm.tqdm(iterable=p_bar, total=len(loader))
            for images, labels in p_bar:
                images = images.cuda()
                label, label_reg = labels[0], labels[1]

                p_label = label[:, 0].cuda()
                y_label = label[:, 1].cuda()

                # Continuous labels
                p_label_reg = label_reg[:, 0].cuda()
                y_label_reg = label_reg[:, 1].cuda()

                pitch, yaw = model(images)

                p_loss = criterion(pitch, p_label)
                y_loss = criterion(yaw, y_label)

                p_pred, y_pred = softmax(pitch), softmax(yaw)

                p_pred = torch.sum(p_pred * idx_tensor, 1) * 4 - 180
                y_pred = torch.sum(y_pred * idx_tensor, 1) * 4 - 180

                p_reg_loss = reg_criterion(p_pred, p_label_reg)
                y_reg_loss = reg_criterion(y_pred, y_label_reg)

                p_loss += p_reg_loss
                y_loss += y_reg_loss

                loss_seq = [p_loss, y_loss]
                grad_seq = [torch.tensor(1.0).cuda() for _ in range(len(loss_seq))]

                if args.distributed:
                    p_loss = utils.reduce_tensor(p_loss, args.world_size)
                    y_loss = utils.reduce_tensor(y_loss, args.world_size)

                avg_loss_pitch.update(p_loss.item(), images.size(0))
                avg_loss_yaw.update(y_loss.item(), images.size(0))

                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    s = ('%10s' * 2 + '%10.3g' + '%10.3g') % (
                        f'{epoch + 1}/{args.epochs}', memory, avg_loss_pitch.avg, avg_loss_yaw.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                last = test(args, model)
                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'MAE': str(f'{last[0]:.3f}'),
                                 'Pitch': str(f'{last[1]:.3f}'),
                                 'Yaw': str(f'{last[2]:.3f}'), })
                log.flush()
                is_best = sum(last) < best

                if is_best:
                    best = sum(last)
                save = {'epoch': epoch, 'model': copy.deepcopy(model).half()}
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
    model.cuda().eval()
    dataset = Dataset(args, True, is_train=False)
    loader = data.DataLoader(dataset, args.batch_size, False, num_workers=4, pin_memory=True)

    softmax = torch.nn.Softmax(dim=1).cuda()

    MAE, total, p_err, y_err = .0, 0, 0, 0
    idx_tensor = [idx for idx in range(args.bin)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()
    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, ('%10s' * 3) % ('MAE', 'Pitch', 'Yaw')):
            images = images.cuda()
            label, label_reg = labels[0], labels[1]
            total += label_reg.size(0)

            p_label = label_reg[:, 0].float() * np.pi / 180
            y_label = label_reg[:, 1].float() * np.pi / 180

            pitch, yaw = model(images)

            # Binned predictions
            # _, pitch_bpred = torch.max(pitch.data, 1)
            # _, yaw_bpred = torch.max(yaw.data, 1)

            # Continuous predictions
            p_pred = softmax(pitch)
            y_pred = softmax(yaw)

            # mapping from binned (0 to 28) to angels (-180 to 180)
            p_pred = torch.sum(p_pred * idx_tensor, 1).cpu() * 4 - 180
            y_pred = torch.sum(y_pred * idx_tensor, 1).cpu() * 4 - 180

            p_pred = p_pred * np.pi / 180
            y_pred = y_pred * np.pi / 180

            for p, y, pl, yl in zip(p_pred, y_pred, p_label, y_label):
                MAE += util.angular(util.gazeto3d([p, y]), util.gazeto3d([pl, yl]))

            p_err += torch.sum(torch.abs((p_pred - p_label) * 180 / np.pi))
            y_err += torch.sum(torch.abs((y_pred - y_label) * 180 / np.pi))
    MAE, p_err, y_err = MAE / total, p_err / total, y_err / total
    print(('%10s' * 3) % (f'{MAE:.3f}', f'{p_err:.3f}', f'{y_err:.3f}'))
    model.float()  # for training
    return MAE, p_err, y_err


def demo(args):
    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = torch.load('./weights/best.pt', 'cuda')
    model = model['model'].float()
    model.cuda().eval()
    detector = RetinaFace(0)
    softmax = torch.nn.Softmax(dim=1).cuda()

    idx_tensor = [idx for idx in range(args.bin)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    transform = transforms.Compose([transforms.Resize(args.inp_size),
                                    transforms.CenterCrop(args.inp_size),
                                    transforms.ToTensor(),
                                    normalize])

    stream = cv2.VideoCapture(-1)

    if not stream.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = stream.read()
        if not ret:
            break
        faces = detector(frame)

        for box, landmarks, score in faces:
            if score < .95:
                continue
            x_min, y_min = int(box[0]), int(box[1])
            x_max, y_max = int(box[2]), int(box[3])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min - int(0.2 * bbox_height))
            y_min = max(0, y_min - int(0.2 * bbox_width))
            x_max = x_max + int(0.2 * bbox_height)
            y_max = y_max + int(0.2 * bbox_width)

            image = frame[y_min:y_max, x_min:x_max, :]
            image = Image.fromarray(image)
            image = image.convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)
            image = image.cuda()

            pitch, yaw = model(image)
            pitch, yaw = softmax(pitch), softmax(yaw)
            pitch = torch.sum(pitch.data * idx_tensor, dim=1) * 4 - 180
            yaw = torch.sum(yaw.data * idx_tensor, dim=1) * 4 - 180

            pitch = pitch.cpu().detach().numpy() * np.pi / 180.0
            yaw = yaw.cpu().detach().numpy() * np.pi / 180.0

            frame = util.draw_gaze(x_min, y_min, bbox_width, bbox_height, frame,
                                   (pitch.item(), yaw.item()), color=(0, 0, 255))

            cv2.imshow("Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../../Datasets/Gaze_360/')
    parser.add_argument('--bin', type=int, default=90)
    parser.add_argument('--arch', type=str, default='18')
    parser.add_argument('--data-name', type=str, default='gaze360')
    parser.add_argument('--angle', type=int, default=180)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--inp-size', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.makedirs('./weights', exist_ok=True)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.demo:
        demo(args)


if __name__ == '__main__':
    main()
