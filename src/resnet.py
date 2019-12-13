"""
resnet 结构网络在当前分类任务中的定制版本
"""
# 实现 ResNet 的基本模块
import torch
from torch import nn
from torch.nn import functional as F


# ResNet 中的基本残差单元
# 通过这里可以看出 pytorch 中允许自定义一个网络子模块，然后指定其 forward 方法，最后可以在主网络中添加该子模块
class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        # 这里的 stride 若在 ResNet18 中传入的为 2，则大概会使图片尺寸成半的减下来
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)   # batch normalization
        # 这一步各参数的设置并不会使图片尺寸发生变化
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        # shortcut  相当与只有通道数发生了改变，图片大小等均没有变化，
        # 且只有输入和输出通道数不一致的情况下才进行了下面的调整（通道数一致的情况下 shortcut 也没必要调整）
        # 用 1x1 的卷积，指定的 stride 进行间隔卷积采样，直接将图片尺寸调整为相同
        if ch_out != ch_in:
            # [b, ch_in, h, w] -> [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortcut
        # extra model : [b, ch_in, h, w] with [b, ch_out, h, w] 这里要考虑到 ch_in 和 ch_out 可能不一样
        # element-wise add:
        out = self.extra(x) + out
        return out


class ResNet18(nn.Module):
    """
    ResNet 的 18 层结构网络的实现
    """
    def __init__(self, num_class):
        super(ResNet18, self).__init__()

        # 注意本次使用的截取后的图片大小为：h x w = 40 x 40
        self.conv1 = nn.Sequential(
            # [b, 3, 40, 40] => [b, 16, 38, 38]
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16)  # batch normalization 维度要与输出维度相同
        )
        # followed 4 blocks
        # padding = 1, kernel_size = 3
        self.blk1 = ResBlk(16, 32, stride=2)    # [b, 16, 38, 38] -> [b, 32, 19, 19]  # 计算非整数向下取整
        self.blk2 = ResBlk(32, 64, stride=2)    # [b, 32, 19, 19] -> [b, 64, 10, 10]
        self.blk3 = ResBlk(64, 128, stride=2)   # [b, 64, 10, 10] -> [b, 128, 5, 5]
        self.blk4 = ResBlk(128, 256, stride=2)  # [b, 128, 5, 5] -> [b, 256, 3, 3]

        # 线性输出层
        self.liner = nn.Linear(256*3*3, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape)

        # flatting
        x = x.view(x.size(0), -1)
        # 线性分类
        x = self.liner(x)

        return x


def main():
    # 测试 ResBlk 模块
    blk = ResBlk(ch_in=64, ch_out=128)  # 仅改变通道数
    tmp = torch.randn(2, 64, 40, 40)
    out = blk(tmp)
    print('block:', out.shape)

    # 测试 ResNet18 模型对输入输出的参数是否能正确匹配
    model = ResNet18(62)     # 指定类别为 62 类
    x = torch.randn(2, 3, 40, 40)
    out = model(x)
    print("resnet:", out.shape)

    # 打印模型中总参数量的大小
    p = sum(map(lambda p: p.numel(), model.parameters()))
    print("parameters size:", p)


if __name__ == '__main__':
    main()

