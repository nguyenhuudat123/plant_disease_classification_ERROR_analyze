import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import json
import time
import pickle
import os
import random
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
import base64

# 1. SIMPLE CNN
# =============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# 2. LENET (BASELINE)
# =============================================================================

class LeNet(nn.Module):
    def __init__(self, num_classes=38):
        super(LeNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(32 * 24 * 24, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# 3. ALEXNET
# =============================================================================

class AlexNet(nn.Module):
    def __init__(self, num_classes=38):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =============================================================================
# 4. RESNET FAMILY
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=38):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, 2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes=38):
        super(ResNet34, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 3, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, 2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, 2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ResNet50(nn.Module):
    def __init__(self, num_classes=38):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(Bottleneck, 64, 3, 1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, 2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, 2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(2048, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# =============================================================================
# 5. MOBILENET V2
# =============================================================================

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=38, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        
        input_channel = 32
        last_channel = 1280
        
        # Building inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        
        # First layer
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        # Last layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(self.last_channel, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# 6. EFFICIENTNET B0
# =============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        else:
            self.expand_conv = None
            expanded_channels = in_channels
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )
        
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, se_channels)
        else:
            self.se = None
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        
        if self.se is not None:
            x = self.se(x)
        
        x = self.project_conv(x)
        
        if self.use_residual:
            x = x + identity
        
        return x


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=38):
        super(EfficientNetB0, self).__init__()
        
        # Building blocks config: (expand_ratio, out_channels, repeats, stride, kernel_size)
        blocks_config = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        self.blocks = nn.ModuleList()
        in_channels = 32
        
        for expand_ratio, out_channels, repeats, stride, kernel_size in blocks_config:
            for i in range(repeats):
                self.blocks.append(
                    MBConvBlock(
                        in_channels, 
                        out_channels, 
                        kernel_size, 
                        stride if i == 0 else 1,
                        expand_ratio
                    )
                )
                in_channels = out_channels
        
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(1280, 512), Swish(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# =============================================================================
# 7. DENSENET121
# =============================================================================

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, 1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False))
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = self.norm1(x)
        new_features = self.relu1(new_features)
        new_features = self.conv1(new_features)
        new_features = self.norm2(new_features)
        new_features = self.relu2(new_features)
        new_features = self.conv2(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
    
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, 1, bias=False))
        self.add_module('pool', nn.AvgPool2d(2, stride=2))
    
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes=38, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet121, self).__init__()
        
        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# =============================================================================
# 9. INCEPTION V3
# =============================================================================

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionV3(nn.Module):
    def __init__(self, num_classes=38):
        super(InceptionV3, self).__init__()
        
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(288, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# =============================================================================
# MODEL DICTIONARY FOR EASY ACCESS
# =============================================================================

# Thêm vào model_lib.py

# =============================================================================
# 10. MOBILENET V3 SMALL
# =============================================================================

class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class HardSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
    
    def forward(self, x):
        out = self.avgpool(x)
        out = F.relu(self.fc1(out))
        out = F.relu6(self.fc2(out) + 3) / 6  # HardSigmoid
        return x * out

class MobileBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, exp_size, se=False, nl='RE'):
        super().__init__()
        self.use_res = stride == 1 and in_ch == out_ch
        activation = nn.ReLU if nl == 'RE' else HardSwish
        
        layers = []
        if exp_size != in_ch:
            layers.extend([
                nn.Conv2d(in_ch, exp_size, 1, bias=False),
                nn.BatchNorm2d(exp_size),
                activation()
            ])
        
        layers.extend([
            nn.Conv2d(exp_size, exp_size, kernel, stride, kernel//2, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            activation()
        ])
        
        self.conv = nn.Sequential(*layers)
        self.se = SEModule(exp_size) if se else None
        self.project = nn.Sequential(
            nn.Conv2d(exp_size, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        out = self.conv(x)
        if self.se:
            out = self.se(out)
        out = self.project(out)
        if self.use_res:
            out = out + x
        return out

class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )
        
        # [in, out, k, s, exp, se, nl]
        cfg = [
            [16, 16, 3, 2, 16, True, 'RE'],
            [16, 24, 3, 2, 72, False, 'RE'],
            [24, 24, 3, 1, 88, False, 'RE'],
            [24, 40, 5, 2, 96, True, 'HS'],
            [40, 40, 5, 1, 240, True, 'HS'],
            [40, 40, 5, 1, 240, True, 'HS'],
            [40, 48, 5, 1, 120, True, 'HS'],
            [48, 48, 5, 1, 144, True, 'HS'],
            [48, 96, 5, 2, 288, True, 'HS'],
            [96, 96, 5, 1, 576, True, 'HS'],
            [96, 96, 5, 1, 576, True, 'HS'],
        ]
        
        layers = []
        for in_ch, out_ch, k, s, exp, se, nl in cfg:
            layers.append(MobileBottleneck(in_ch, out_ch, k, s, exp, se, nl))
        self.bottlenecks = nn.Sequential(*layers)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HardSwish()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =============================================================================
# 11. SHUFFLENET V2
# =============================================================================

def channel_shuffle(x, groups):
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        
        branch_channels = out_channels // 2
        
        if stride == 1:
            self.branch1 = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU()
            )
        
        in_ch = in_channels if stride > 1 else branch_channels
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, 3, stride, 1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=38, width_mult=1.0):
        super().__init__()
        
        # [out_channels, num_blocks, stride]
        stage_out_channels = [24, 116, 232, 464, 1024]
        stage_out_channels = [int(c * width_mult) for c in stage_out_channels]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, stage_out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(stage_out_channels[0]),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Stage 2
        self.stage2 = self._make_stage(stage_out_channels[0], stage_out_channels[1], 4)
        # Stage 3
        self.stage3 = self._make_stage(stage_out_channels[1], stage_out_channels[2], 8)
        # Stage 4
        self.stage4 = self._make_stage(stage_out_channels[2], stage_out_channels[3], 4)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(stage_out_channels[3], stage_out_channels[4], 1, bias=False),
            nn.BatchNorm2d(stage_out_channels[4]),
            nn.ReLU()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(stage_out_channels[4], num_classes)
        )
    
    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [ShuffleBlock(in_channels, out_channels, 2)]
        for _ in range(num_blocks - 1):
            layers.append(ShuffleBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =============================================================================
# UPDATE get_all_models()
# =============================================================================

def get_all_models(num_classes=38):
    """Return dictionary of all available models"""
    return {
        # Basic Models
        'LeNet': LeNet(num_classes),
        'SimpleCNN': SimpleCNN(num_classes),
        'AlexNet': AlexNet(num_classes),
        
        # ResNet Family  
        'ResNet18': ResNet18(num_classes),
        'ResNet34': ResNet34(num_classes),
        'ResNet50': ResNet50(num_classes),
        
        # Modern Efficient Models
        'MobileNetV2': MobileNetV2(num_classes),
        'MobileNetV3Small': MobileNetV3Small(num_classes),  # NEW
        'ShuffleNetV2': ShuffleNetV2(num_classes),           # NEW
        'EfficientNetB0': EfficientNetB0(num_classes),
        'DenseNet121': DenseNet121(num_classes),
        
        # Advanced Models
        'InceptionV3': InceptionV3(num_classes),
    }

def get_model_info():
    """Return information about each model"""
    return {
        'LeNet': {'type': 'Basic CNN', 'params': '~0.5M', 'description': 'Classic CNN baseline'},
        'SimpleCNN': {'type': 'Basic CNN', 'params': '~6M', 'description': 'Simple deep CNN'},
        'AlexNet': {'type': 'Classic CNN', 'params': '~25M', 'description': 'Historic breakthrough model'},
        
        'ResNet18': {'type': 'ResNet', 'params': '~11M', 'description': 'Lightweight residual network'},
        'ResNet34': {'type': 'ResNet', 'params': '~21M', 'description': 'Medium residual network'},
        'ResNet50': {'type': 'ResNet', 'params': '~25M', 'description': 'Deep residual with bottlenecks'},
        
        'MobileNetV2': {'type': 'Efficient', 'params': '~3.5M', 'description': 'Mobile-optimized architecture'},
        'EfficientNetB0': {'type': 'Efficient', 'params': '~5M', 'description': 'Compound scaled efficient model'},
        'DenseNet121': {'type': 'Dense', 'params': '~8M', 'description': 'Dense connectivity patterns'},
        
        'InceptionV3': {'type': 'Inception', 'params': '~27M', 'description': 'Multi-scale feature extraction'},
    }