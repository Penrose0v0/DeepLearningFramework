import torch
import torch.nn as nn

"""2D UNet"""
class DoubleConv(nn.Sequential):
    def __init__(self, in_c, out_c, mid_c=None):
        if mid_c is None:
            mid_c = out_c
        super().__init__(
            nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )


class Down(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_c, out_c)
        )


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c // 2, in_c // 2, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv = DoubleConv(in_c, out_c, in_c // 2)

    def forward(self, cur, prev):
        cur = self.up(cur)
        cur = torch.cat([cur, prev], dim=1)
        return self.conv(cur)


class UNet(nn.Module):
    def __init__(self, in_c, out_c, base_c=64, num=4):
        super().__init__()
        self.in_conv = DoubleConv(in_c, base_c)  # 3->64
        self.down = [Down(base_c * (2 ** i), base_c * (2 ** (i + 1))) for i in range(num - 1)]
        self.down.append(Down(base_c * (2 ** (num - 1)), base_c * (2 ** (num - 1))))

        self.up = [Up(base_c * (2 ** (i + 1)), base_c * (2 ** (i - 1))) for i in range(num - 1, 0, -1)]
        self.up.append(Up(base_c * 2, base_c))

        self.out_conv = nn.Conv2d(base_c, out_c, kernel_size=1)

    def encode(self, x):
        cur = self.in_conv(x)
        prev_list = []

        for down in self.down:
            prev_list.append(cur)
            cur = down(cur)

        return cur, prev_list

    def decode(self, cur, prev_list):
        for prev, up in zip(prev_list[::-1], self.up):
            cur = up(cur, prev)

        return self.out_conv(cur)

    def forward(self, x):
        return self.decode(*self.encode(x))


"""ResNet"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        cur = self.relu(self.bn1(self.conv1(x)))
        cur = self.bn2(self.conv2(cur))
        print(cur.shape, identity.shape)
        if self.downsample is not None:
            identity = self.downsample(x)

        cur += identity
        cur = self.relu(cur)

        return cur


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1, downsample=None, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(out_c * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(in_c, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, groups=groups,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_c * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        cur = self.relu(self.bn1(self.conv1(x)))
        cur = self.relu(self.bn2(self.conv2(cur)))
        cur = self.bn3(self.conv3(cur))

        if self.downsample is not None:
            identity = self.downsample(x)

        cur += identity
        cur = self.relu(cur)

        return cur

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, base_width=64):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, self.in_c, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_c)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.include_top = include_top
        self.groups = groups
        self.base_width = base_width

        layers = []
        for i, cn in enumerate(zip([64, 128, 256, 512], blocks_num)):
            c, n = cn
            s = 1 if i == 0 else 2
            layers.append(self._make_layer(block, c, n, s))
        self.layers = nn.Sequential(*layers)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_c != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = [
            block(self.in_c, channel,
                  downsample=downsample, stride=stride, groups=self.groups, base_width=self.base_width)
        ]
        self.in_c = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(self.in_c, channel, groups=self.groups, base_width=self.base_width)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layers(x)

        if self.include_top:
            x = self.avgpool(x)
            x = self.fc(torch.flatten(x, 1))

        return x

def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


if __name__ == "__main__":
    # net = UNet(in_c=3, out_c=3, num=4)
    net = resnet101()

    inp = torch.rand(1, 3, 512, 512)
    out = net(inp)
    print(out.shape)
