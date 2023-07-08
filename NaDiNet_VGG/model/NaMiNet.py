import torch
import torch.nn.functional as F
from torchvision import models
import torch
import torch.nn as nn
from model.from_origin import Backbone_VGG16_in3


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class MIB(nn.Module):
    def __init__(self, inc1, inc2):
        super(MIB, self).__init__()
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mid = (inc1 + inc2) // 2
        self.cbam1 = CBAMBlock(self.mid)
        self.cbam2 = CBAMBlock(self.mid)
        self.cbam3 = CBAMBlock(self.mid)
        self.cbam4 = CBAMBlock(self.mid)
        self.conv1 = nn.Sequential(nn.Conv2d(2 * self.mid, 2 * self.mid, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(2 * self.mid),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(2 * self.mid, 2 * self.mid, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(2 * self.mid),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.mid * 2, self.mid, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.mid),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(self.mid * 2, self.mid, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.mid),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(self.mid * 2, self.mid, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.mid),
                                   nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(self.mid * 2, self.mid, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.mid),
                                   nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(self.mid, inc1, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(inc1),
                                   nn.ReLU(inplace=True))

    def forward(self, x, y):
        f1 = x
        f2 = y
        if x.size()[2] == y.size()[2]:
            f1 = torch.cat((x, y), dim=1)
            f2 = torch.cat((self.down2(x), self.down2(y)), dim=1)
        elif x.size()[2] == 2 * y.size()[2]:
            f1 = torch.cat((x, self.up2(y)), dim=1)
            f2 = torch.cat((self.down2(x), y), dim=1)
        else:
            raise Exception("wrong input size!")
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        x1, x2 = torch.split(f1, [self.mid, self.mid], dim=1)
        x3, x4 = torch.split(f2, [self.mid, self.mid], dim=1)
        z1 = self.cbam1(x1)
        z2 = self.cbam2(self.conv3(torch.cat((z1, x2), dim=1)))
        z3 = self.cbam3(x3)
        z4 = self.cbam4(self.conv4(torch.cat((z3, x4), dim=1)))
        rst1 = self.conv5(torch.cat((z1, z2), dim=1))
        rst2 = self.conv6(torch.cat((z3, z4), dim=1))

        rst2 = self.up2(rst2)

        return self.conv7(torch.mul(rst1, rst2))


class NAM(nn.Module):
    def __init__(self):
        super(NAM, self).__init__()
        self.sigma = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = x.view(b, c, -1)
        k = x.view(b, c, -1).permute(0, 2, 1)
        dot = torch.bmm(q, k)

        q_ = torch.norm(q, p=2, dim=2).view(b, -1, c).permute(0, 2, 1)
        k_ = torch.norm(k, p=2, dim=1).view(b, -1, c)
        dot_ = torch.bmm(q_, k_) + 1e-08
        atte_map = torch.div(dot, dot_)
        v = x.view(b, c, -1)
        out = torch.bmm(atte_map, v)
        out = out.view(b, c, h, w)

        return self.sigma * out + x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NaMiNet(nn.Module):
    def __init__(self):
        super(NaMiNet, self).__init__()

        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        self.reduce1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.reduce2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.reduce3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.reduce4 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))
        self.reduce5 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.NAM1 = NAM()
        self.NAM2 = NAM()
        self.NAM3 = NAM()
        self.NAM4 = NAM()
        self.NAM5 = NAM()

        # -------------Bridge--------------

        self.GCM = GCM(128, 128)

        # -------------Decoder--------------

        self.decoder5 = MIB(128, 128)
        self.decoder4 = MIB(128, 128)
        self.decoder3 = MIB(128, 128)
        self.decoder2 = MIB(128, 128)
        self.decoder1 = MIB(128, 128)

        # ---------------sampling----------------
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # -------------Side Output--------------
        self.outconvb = nn.Conv2d(128, 1, 1, stride=1)
        self.outconv5 = nn.Conv2d(128, 1, 1, stride=1)
        self.outconv4 = nn.Conv2d(128, 1, 1, stride=1)
        self.outconv3 = nn.Conv2d(128, 1, 1, stride=1)
        self.outconv2 = nn.Conv2d(128, 1, 1, stride=1)
        self.outconv1 = nn.Conv2d(128, 1, 1, stride=1)

    def forward(self, x):
        hx = x

        # -------------Encoder-------------
        h1 = self.encoder1(hx)
        h2 = self.encoder2(h1)
        h3 = self.encoder4(h2)
        h4 = self.encoder8(h3)
        h5 = self.encoder16(h4)

        # -------------Channel reduce-------------
        h1 = self.reduce1(h1)  # 96x96x64  -> ..128
        h2 = self.reduce2(h2)  # 48x48x128 -> ..128
        h3 = self.reduce3(h3)  # 24x24x256 -> ..128
        h4 = self.reduce4(h4)  # 12x12x896 -> ..128
        h5 = self.reduce5(h5)  # 12x12x1920-> ..128

        # -------------Bridge-------------
        hbg = self.GCM(h5)  # 12x12x128

        # -------------CAM-------------
        h1 = self.NAM1(h1)
        h2 = self.NAM2(h2)
        h3 = self.NAM3(h3)
        h4 = self.NAM4(h4)
        h5 = self.NAM5(h5)

        # -------------Decoder with MIB-------------

        hd5 = self.decoder5(h5, hbg)
        hd4 = self.decoder4(h4, hd5)
        hd3 = self.decoder3(h3, hd4)
        hd2 = self.decoder2(h2, hd3)
        hd1 = self.decoder1(h1, hd2)

        # -------------Side Output-------------
        db = self.outconvb(hbg)
        db = self.up16(db)  # 11 -> 352

        d5 = self.outconv5(hd5)
        d5 = self.up16(d5)  # 11 -> 352

        d4 = self.outconv4(hd4)
        d4 = self.up8(d4)  # 11 -> 352

        d3 = self.outconv3(hd3)
        d3 = self.up4(d3)  # 22 -> 352

        d2 = self.outconv2(hd2)
        d2 = self.up2(d2)  # 44 -> 352

        d1 = self.outconv1(hd1)

        return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(db)
