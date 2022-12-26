import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


class MapEncoder(nn.Module):
    def __init__(self, map_size, input_channel, output_channel):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 8, stride=2, padding=3),   # 100 -> 50
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=output_channel),
            nn.ReLU(inplace=True),
        )

        cnn_dims = np.array([map_size, map_size], dtype=np.float32)
        self._cnn_layers_kernel = [(8, 8), (5, 5), (3, 3)]
        self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        self._cnn_layers_padding = [(3, 3), (1, 1), (1, 1)]
        for kernel, stride, padding in zip(self._cnn_layers_kernel, self._cnn_layers_stride, self._cnn_layers_padding):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array(padding, dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel=np.array(kernel, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )
        
        self.output_shape = [output_channel, cnn_dims[0], cnn_dims[1]]

    def _conv_output_dim(self, dimension, padding, dilation, kernel, stride):
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, rgb_map):
        return self.cnn(rgb_map)


class MapDecoder(nn.Module):
    def __init__(self, n_channel_in):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up0 = convrelu(64 + 64, 128, 3, 1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.output_shape = [64, 100, 100]

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)

        layer1 = self.layer1_1x1(layer1)
        x = self.upsample(layer1)
        
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        return x
