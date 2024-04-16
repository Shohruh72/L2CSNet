import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms as T


class Dataset(data.Dataset):
    def __init__(self, data_dir, angle=180, is_train=True):
        self.angle = angle if is_train else 90
        self.bin_width = 4
        self.is_train = is_train
        self.data_dir = data_dir
        self.lines = []

        self.transform = T.Compose([T.Resize(448), T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        label_file = 'train.label' if is_train else 'test.label'
        with open(os.path.join(data_dir, 'Label', label_file)) as f:
            lines = f.readlines()
            lines.pop(0)
            self.list_len = len(lines)
            for line in lines:
                gaze2d = line.strip().split(" ")[5]
                label = np.array(gaze2d.split(",")).astype("float")
                if all(abs(label * 180 / np.pi) <= self.angle):
                    self.lines.append(line)

        print("{} items removed from dataset that have an angle > {}".format(self.list_len - len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split(" ")
        face = line[0]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi

        img = Image.open(os.path.join(self.data_dir, 'Image', face))
        # if self.is_train:
        img = self.transform(img)

        bins = np.array(range(-1 * self.angle, self.angle, self.bin_width))

        labels = np.digitize([pitch, yaw], bins) - 1
        cont_labels = torch.FloatTensor([pitch, yaw])

        return img, [labels, cont_labels]
