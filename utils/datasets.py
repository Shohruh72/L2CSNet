import numpy as np
from PIL import Image
from os.path import join

import torch
from torch.utils import data
from torchvision import transforms as T


class Dataset(data.Dataset):
    def __init__(self, args, augment=True, is_train=True):
        self.args = args
        self.success_rate = 0
        self.is_train = is_train
        self.augment = augment
        self.angle = self.args.angle

        self.samples = self.load_label(self.args.data_dir, self.args.angle, self.args.data_name, is_train)

        self.transforms = T.Compose([T.Resize(448), T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        line = self.samples[idx]
        line = line.strip().split(" ")
        face, name = line[0], line[3]
        if self.args.data_name == 'gaze360':
            gaze = line[5]
        else:
            gaze = line[7]
        label = np.array(gaze.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.float)

        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi

        image = Image.open(join(self.args.data_dir, 'Image', face))

        if self.augment:
            image = self.transforms(image)

        bins = np.array(range(-1 * self.args.angle, self.args.angle, 4))
        labels = np.digitize([pitch, yaw], bins) - 1
        cont_labels = torch.FloatTensor([pitch, yaw])

        return image, [labels, cont_labels]

    @staticmethod
    def load_label(data_dir, angle, data_name, is_train):
        samples = []
        label_file = 'train.label' if is_train else 'test.label'
        if not is_train:
            angle = 90
        with open(join(data_dir, 'Label', label_file), 'r') as f:
            lines = f.readlines()
            lines.pop(0)
            for line in lines:
                if data_name == 'gaze360':
                    gaze = line.strip().split(" ")[5]
                elif data_name == 'mpiigaze':
                    gaze = line.strip().split(" ")[7]
                label = np.array(gaze.split(",")).astype("float")
                if abs((label[0] * 180 / np.pi)) <= angle and abs((label[1] * 180 / np.pi)) <= angle:
                    samples.append(line)
        return samples
