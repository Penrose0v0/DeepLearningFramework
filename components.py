import torch
import torch.nn as nn
import numpy as np

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

        down = [Down(base_c * (2 ** i), base_c * (2 ** (i + 1))) for i in range(num - 1)]
        down.append(Down(base_c * (2 ** (num - 1)), base_c * (2 ** (num - 1))))
        self.down = nn.ModuleList(down)

        up = [Up(base_c * (2 ** (i + 1)), base_c * (2 ** (i - 1))) for i in range(num - 1, 0, -1)]
        up.append(Up(base_c * 2, base_c))
        self.up = nn.ModuleList(up)

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


"""Transformer"""
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_causal_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.w_q = nn.Linear(embed_dim, d_k * num_heads, bias=False)
        self.w_k = nn.Linear(embed_dim, d_k * num_heads, bias=False)
        self.w_v = nn.Linear(embed_dim, d_v * num_heads, bias=False)
        self.fc = nn.Linear(num_heads * d_v, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def ScaledDotProductAttention(self, q, k, v, attn_mask=None):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores.masked_fill_(attn_mask.bool(), -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

    def forward(self, q, k, v, attn_mask=None):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = q, q.shape[0]
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        context, attn = self.ScaledDotProductAttention(q, k, v, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_v)

        output = self.dropout(self.fc(context))
        output = self.layer_norm(output + residual)
        return output, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_dim, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, embed_dim, bias=False))
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        output = self.dropout(self.fc(x))
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, d_k, d_v, num_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, x, attn_mask=None):
        y, attn = self.attention(x, x, x, attn_mask)
        y = self.ffn(y)
        return y, attn

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, d_k, d_v, d_ff, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x):
        # x = ... x positional embedding
        attn_mask = get_attn_pad_mask(x, x).to(x.device)
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attn_list.append(attn)
        return x, attn_list


if __name__ == "__main__":
    # net = UNet(in_c=3, out_c=3, num=4)
    net = resnet101()

    inp = torch.rand(1, 3, 512, 512)
    out = net(inp)
    print(out.shape)
