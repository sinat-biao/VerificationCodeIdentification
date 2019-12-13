"""
加载验证码数据集（加载的是切分后单个字符的图片）
"""
import torch
import os
import glob
import random
import csv
# 自定义数据集类需继承的父类
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class VerificationCode(Dataset):
    # root 表示数据集存储路径
    # resize 表示加载进来的图片的 size（这里是为了保证处理图片大小一致）
    # mode 表示当前数据是做什么模式（training or test）
    def __init__(self, root, resize, mode):
        super(VerificationCode, self).__init__()
        self.root = root        # 注意这里的 root 是数据集文件的 root（根目录）
        self.resize = resize    # 指定需要重新调整到的尺寸的大小

        # 设置字典指定映射关系
        self.name2label = {}    # 如 0 -> 0; cA -> 11;
        # 为每一个图片集做映射（其实就是打标签），这里注意要使图片和标签好好对应，后面不能发生变化
        # 遍历当前路径下的所有文件夹
        # 由于 listdir 返回的 list 顺序不固定，所以还需手动的排序一下
        for name in sorted(os.listdir(os.path.join(root))):
            # 只取文件夹
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        print(self.name2label)
        # {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        # 'cA': 10, 'cB': 11, 'cC': 12, 'cD': 13, 'cE': 14, 'cF': 15, 'cG': 16, 'cH': 17, 'cI': 18, 'cJ': 19, 'cK': 20, 'cL': 21, 'cM': 22, 'cN': 23, 'cO': 24, 'cP': 25, 'cQ': 26, 'cR': 27, 'cS': 28, 'cT': 29, 'cU': 30, 'cV': 31, 'cW': 32, 'cX': 33, 'cY': 34, 'cZ': 35,
        # 'la': 36, 'lb': 37, 'lc': 38, 'ld': 39, 'le': 40, 'lf': 41, 'lg': 42, 'lh': 43, 'li': 44, 'lj': 45, 'lk': 46, 'll': 47, 'lm': 48, 'ln': 49, 'lo': 50, 'lp': 51, 'lq': 52, 'lr': 53, 'ls': 54, 'lt': 55, 'lu': 56, 'lv': 57, 'lw': 58, 'lx': 59, 'ly': 60, 'lz': 61}

        # image, label
        self.images, self.labels = self.load_csv('images_train.csv')

        # 由于训练和测试使用的样本不一样，所以这里需要做个判断
        if mode == 'train':     # 60%
            # 取前 60% 的数据用来作为训练数据
            self.images = self.images[: int(0.6*len(self.images))]
            self.labels = self.labels[: int(0.6*len(self.labels))]
        elif mode == 'val':     # 20%
            # 取中间 20% 作为验证集
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else:                   # 20%
            # 取最后 20% 作为测试集
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]

    def load_csv(self, filename):
        """
        理想情况下是将图片数据以键值对： image + label 的形式保存到 csv 文件中。
        但是一次加载这么多的图片肯定会爆内存，所以通过：image_path + label 的形式保存。
        :param filename:
        :return:
        """
        # 如果当前目录下不存在该 csv 文件，才遍历文件并创建 csv 文件
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
            # 1167, 'E:/AllDateSets/VerificationCodeDataSet/subimg/train_data\\0\\1.jpg'
            print(len(images), images)
            # 将对应关系保存到 csv 文件中
            # 首先将顺序打乱（随机化）
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)  # csv 形式比较简单，就是每一行用逗号分开
                for img in images:
                    # os.seq 会自动的匹配 windows 和 linux 下不同的路径分隔符进而将其分开
                    name = img.split(os.sep)[-2]    # -2 刚好是类别名
                    label = self.name2label[name]
                    # 每一行写入形式：'E:/AllDateSets/VerificationCodeDataSet/subimg/train_data\\0\\1.jpg', 1
                    writer.writerow([img, label])
            print('writen into csv file:', filename)

        # 从 csv 文件中读数据（键值对）
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        # print(len(images), images)
        # print(len(labels), labels)
        assert len(images) == len(labels)   # 判断图片与标签的数目是否一致，若不一致，会抛出异常
        return images, labels

    def __len__(self):
        # 由于在 __init__ 中实现了按 train、valid、test 分配数据，所以这里返回的 len 是分配之后的长度
        return len(self.images)

    def denormalize(self, x_hat):
        """逆正则化的过程，主要是保证图片再显示的时候不会出现颜色上的问题"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 正常的正则化计算过程：x_hat = (x-mean)/std
        # 所以有：x = x_hat*std + mean
        # 前提是必须保证上面参与运算的 x, x_hat, mean, std 的 shape 一致
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)     # unsqueeze(1) 表示在后面插入一个维度
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean
        return x

    # 返回当前 idx 位置上的元素值（这里需要返回的是一张图片以及该图片所对应的一个 label）
    # 这一部分函数主要是在 DataLoader 加载原始图片数据之后然后送入神经网络之前调用，保证送入神经网络的图片数据是增强过的
    # 也就是说一开始图片为 40*40 ，就找这个尺寸读入模型，然后模型自己内部进行 resize
    def __getitem__(self, idx):
        # idx: [0~len(images)]
        # self.images, self.labels
        # 首先取出图片的路径和标签
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            # 首先通过路径读入图片数据
            lambda x: Image.open(x).convert('RGB'),
            # 将图片 resize 成指定的 size
            # 由于接下来需要使用旋转等数据增强手段，所以这里将图片 resize 适当放大
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            # 数据增强
            transforms.RandomRotation(15),  # 随机旋转 15 度
            # 转成 tensor 数据类型
            transforms.ToTensor(),
            # 归一化（其中的 mean 和 std 是统计的 imagenet 中所有图片得到的值）
            # 这样一来，我们的数据中的每个值就变成了[-1,1]的数了
            # 这样可以保证所有的图像分布都相似，也就是在训练的时候更容易收敛，也就是训练的更快更好了
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            # 但是 normalize 之后得到的图像由于值都变为 -1~1 了，颜色必然也会发生变化，所以显示就会很奇怪，这时就需要手动的取逆正则化一下再显示
        ])
        img = tf(img)
        label = torch.tensor(label)      # 不要忘了将 label 也转成 tensor 数据类型
        return img, label


def main():
    # 验证 getitem 函数
    # 首先引入可视化模块，方便检查
    import visdom
    import time

    viz = visdom.Visdom()

    # 初始化一个 VerificationCode 实例（会生成 csv 文件，并依传入的模式返回指定比例的数据）
    # 并且指定输入图片的尺寸 reshape 成 40*40
    db = VerificationCode('E:/AllDateSets/VerificationCodeDataSet/subimg/train_data', 40, 'train')

    # 获取其中一张图片及其对应的 label
    x, y = next(iter(db))   # iter(db) 获得 db 的迭代器，next() 函数获取迭代器中的一组元素
    print('sample:', x.shape, y.shape, y)
    # 显示图片（记得将正则化的 x 逆正则化一下再显示）
    viz.image(db.denormalize(x), win='simple', opts=dict(title='simple_x'))

    # 每次读取一批数据了
    loader = DataLoader(db, batch_size=32, shuffle=True)
    for x, y in loader:
        # print(x.shape, y.shape)     # torch.Size([32, 3, 50, 50]) torch.Size([32]) 说明这里加载出来的数据已经数据增强过了
        # nrow = 8 表示每行显示 8 个
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(5)


if __name__ == '__main__':
    main()
