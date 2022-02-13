import torch
import torch.nn.functional as F
import torchvision.models
import collections
import math

import torch.nn as nn


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
class BottleNeck(torch.nn.Module):
    expansion=4
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BackBone(torch.nn.Module):
        def __init__(self):
            super(BackBone, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(BottleNeck, 64, 3)
            self.layer2 = self._make_layer(BottleNeck, 128, 4, stride=2)
            self.layer3 = self._make_layer(BottleNeck, 256, 6, stride=2)
            self.layer4 = self._make_layer(BottleNeck, 512, 3, stride=2)

        def _make_layer(self, block, planes, block_nums, stride=1):
            downsample = None
            outplanes = planes * block.expansion
            if stride != 1 or self.inplanes != outplanes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, outplanes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(outplanes)
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = outplanes
            for i in range(1, block_nums):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)


        def forward(self, x):
            x = self.conv1(x)

            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x


class Decoder(nn.Module):

    def __init__(self,output_size):
        super(Decoder, self).__init__()
        self.output_size=output_size
        self.conv2 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bilinear = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x=self.conv2(x)
        x=self.bn2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x= self.conv3(x)
        x=self.bilinear(x)
        return x
class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, num_channels, stride=2):
        super(Unpool, self).__init__()

        self.num_channels = num_channels
        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.weights = torch.autograd.Variable(torch.zeros(num_channels, 1, stride, stride).cuda())
        self.weights[:,:,0,0] = 1

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, stride=self.stride,groups=self.num_channels)
class UpConv(Decoder):
    def UpConvModule(self,in_channels):
        return nn.Sequential(collections.OrderedDict([
            ('unpool',Unpool(in_channels)),
            ('conv', nn.Conv2d(in_channels, in_channels // 2, kernel_size=5, stride=1, padding=2, bias=False)),
            ('batchnorm', nn.BatchNorm2d(in_channels // 2)),
            ('relu', nn.ReLU()),
        ]))
    def __init__(self,output_size):
        super(UpConv, self).__init__(output_size)

        self.layer1=self.UpConvModule(1024)
        self.layer2 = self.UpConvModule(512)
        self.layer3 = self.UpConvModule(256)
        self.layer4 = self.UpConvModule(128)
class UpProj(Decoder):

    class UpProjModule(nn.Module):
        # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
        #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
        #   bottom branch: 5*5 conv -> batchnorm

        def __init__(self, in_channels):
            super(UpProj.UpProjModule, self).__init__()
            out_channels = in_channels//2
            self.unpool = Unpool(in_channels)
            self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('conv1',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm1', nn.BatchNorm2d(out_channels)),
              ('relu',      nn.ReLU()),
              ('conv2',      nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)),
              ('batchnorm2', nn.BatchNorm2d(out_channels)),
            ]))
            self.bottom_branch = nn.Sequential(collections.OrderedDict([
              ('conv',      nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.unpool(x)
            x1 = self.upper_branch(x)
            x2 = self.bottom_branch(x)
            x = x1 + x2
            x = self.relu(x)
            return x

    def __init__(self, output_size):
        super(UpProj, self).__init__(output_size)
        self.layer1 = self.UpProjModule(1024)
        self.layer2 = self.UpProjModule(512)
        self.layer3 = self.UpProjModule(256)
        self.layer4 = self.UpProjModule(128)


class FCRN(nn.Module):
    def __init__(self,output_size):
        """

        :param output_size: the models final ouput shape  (we use (228,304))

        """
        super(FCRN, self).__init__()

        self.backbone=BackBone()
        self.decoder=UpProj(output_size)

        self.init_backbone()
        self.decoder.apply(weights_init)

    def init_backbone(self):
        backbone_dict=self.backbone.state_dict()

        resnet = torchvision.models.resnet50()
        resnet.load_state_dict(torch.load('resnet50-19c8e357.pth'))
        resnet50_pretrained_dict = resnet.state_dict()
        resnet50_pretrained_dict = {k: v for k, v in resnet50_pretrained_dict.items() if k in backbone_dict}
        backbone_dict.update(resnet50_pretrained_dict)
        self.backbone.load_state_dict(backbone_dict)

        print("resnet50_pretrained_dict loaded.")



    def forward(self,x):
        x=self.backbone(x)
        x=self.decoder(x)
        return  x

