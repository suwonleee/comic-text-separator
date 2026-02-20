# GPL-3.0 — Detection backbone (ResNet34 + FPN + DBHead)
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torchvision.models import resnet34


class SpatialAttention(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.attn = nn.MultiheadAttention(planes, 8)

    def forward(self, x):
        res = x
        n, c, h, w = x.shape
        x = einops.rearrange(x, 'n c h w -> (h w) n c')
        x = self.attn(x, x, x)[0]
        x = einops.rearrange(x, '(h w) n c -> n c h w', n=n, c=c, h=h, w=w)
        return res + x


class DownBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.down = nn.AvgPool2d(2, stride=2) if stride > 1 else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DBHead(nn.Module):
    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 4, 2, 1),
        )
        self.binarize.apply(self._init_weights)
        self.thresh = self._build_thresh(in_channels)
        self.thresh.apply(self._init_weights)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = torch.reciprocal(1 + torch.exp(-self.k * (shrink_maps.sigmoid() - threshold_maps)))
            return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return torch.cat((shrink_maps, threshold_maps), dim=1)

    @staticmethod
    def _init_weights(m):
        if m.__class__.__name__.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif m.__class__.__name__.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _build_thresh(self, inner_channels):
        self.thresh = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 4, 2, 1),
            nn.Sigmoid(),
        )
        return self.thresh


class DBNetModel(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.backbone = resnet34(pretrained=True if pretrained else False)
        self.conv_db = DBHead(64, 0)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )
        self.down_conv1 = DownBlock(0, 512, 512, 2)
        self.down_conv2 = DownBlock(0, 512, 512, 2)
        self.down_conv3 = DownBlock(0, 512, 512, 2)
        self.upconv1 = UpBlock(0, 512, 256)
        self.upconv2 = UpBlock(256, 512, 256)
        self.upconv3 = UpBlock(256, 512, 256)
        self.upconv4 = UpBlock(256, 512, 256)
        self.upconv5 = UpBlock(256, 256, 128)
        self.upconv6 = UpBlock(128, 128, 64)
        self.upconv7 = UpBlock(64, 64, 64)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        h4 = self.backbone.layer1(x)
        h8 = self.backbone.layer2(h4)
        h16 = self.backbone.layer3(h8)
        h32 = self.backbone.layer4(h16)
        h64 = self.down_conv1(h32)
        h128 = self.down_conv2(h64)
        h256 = self.down_conv3(h128)
        up256 = self.upconv1(h256)
        up128 = self.upconv2(torch.cat([up256, h128], dim=1))
        up64 = self.upconv3(torch.cat([up128, h64], dim=1))
        up32 = self.upconv4(torch.cat([up64, h32], dim=1))
        up16 = self.upconv5(torch.cat([up32, h16], dim=1))
        up8 = self.upconv6(torch.cat([up16, h8], dim=1))
        up4 = self.upconv7(torch.cat([up8, h4], dim=1))
        return self.conv_db(up8), self.conv_mask(up4)
