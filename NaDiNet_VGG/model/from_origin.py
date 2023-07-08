# -*- coding: utf-8 -*-
from torch import nn

from model.vgg import vgg16_bn


def Backbone_VGG16_in3():
    net = vgg16_bn(pretrained=True, progress=True)
    div_1 = nn.Sequential(*list(net.children())[0][0:6])
    div_2 = nn.Sequential(*list(net.children())[0][6:13])
    div_4 = nn.Sequential(*list(net.children())[0][13:23])
    div_8 = nn.Sequential(*list(net.children())[0][23:33])
    div_16 = nn.Sequential(*list(net.children())[0][33:43])
    return div_1, div_2, div_4, div_8, div_16
