import math
import copy

import torch
import torchvision.models.resnet as resnet
import torch.utils.model_zoo as model_zoo


class L2CS(torch.nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_gaze = torch.nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = torch.nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = torch.nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # gaze
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze


class EMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


def load_weights(model, ckpt):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in ckpt.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
    return model


def gaze_net(model_name, bins):
    if model_name == '18':
        model = L2CS(resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        return load_weights(model, model_zoo.load_url(pre_url))
    elif model_name == 'ResNet34':
        model = L2CS(resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        return load_weights(model, model_zoo.load_url(pre_url))
    elif model_name == 'ResNet101':
        model = L2CS(resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        return load_weights(model, model_zoo.load_url(pre_url))
    elif model_name == 'ResNet152':
        model = L2CS(resnet.Bottleneck, [3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
        return load_weights(model, model_zoo.load_url(pre_url))
    else:
        if model_name != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        return load_weights(model, model_zoo.load_url(pre_url))