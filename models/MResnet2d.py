import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch





def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=(stride, 1),
                     padding=(1, 0), bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=(stride, 1), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, out_channel=10, mode='MSE', zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.layer1 = self._make_layer(block, 16, layers[0], 2)
        self.layer2 = self._make_layer(block, 16, layers[1], 2) #16
        self.layer3 = self._make_layer(block, 32, layers[2])  #32
        #self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 14))
        cat_number = 32*2*14*4
        self.fc = nn.Sequential(nn.Linear(cat_number * block.expansion, 100), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(100, out_channel))
        """
        self.fc1 = nn.Sequential(nn.Linear(cat_number * block.expansion, 100), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout_rate), nn.Linear(100, out_channel), nn.ReLU(inplace=True),)
        """
        self.fc2 = nn.Sequential(nn.Linear(100, out_channel))
        self.fc3 = nn.Sequential(nn.Linear(100, out_channel))
        self.fc_std = nn.Sequential(nn.Linear(100, out_channel), nn.Softplus())
        self.mode = mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        #x = self.layer4(x)

        #x = self.avgpool(x)
        #x_max = self.maxpool(x)
        #x = torch.cat((x_avg, x_max), 1)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        if self.mode == 'MSE':
            x = self.fc(x)
            x = self.dropout(x)
            x = self.fc1(x)
            return x
        elif self.mode == 'QL':
            x = self.fc(x)
            x = self.dropout(x)
            x1 = self.fc1(x)
            x2 = self.fc2(x)
            x3 = self.fc3(x)
            return x1, x2, x3
        elif self.mode == 'GD':
            x = self.fc(x)
            x = self.dropout(x)
            x1 = self.fc1(x)
            x2 = self.fc_std(x)
            return x1, x2
        else:
            x = self.fc1(x)
            return x


def Mresnet(**kwargs):
    """Constructs a modified ResNet model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


