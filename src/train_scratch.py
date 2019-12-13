"""
训练模型
"""
import torch
import torchvision
from torch import optim, nn
import visdom
from torch.utils.data import DataLoader
from verificationcode import VerificationCode
from resnet import ResNet18

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20

device = torch.device('cuda')
# 设置随机种子，以便能更好的复现
torch.manual_seed(1234)

# 导入数据（设置 db 对象）
train_db = VerificationCode('E:/AllDateSets/VerificationCodeDataSet/subimg/train_data', 40, mode='train')
val_db = VerificationCode('E:/AllDateSets/VerificationCodeDataSet/subimg/train_data', 40, mode='val')
test_db = VerificationCode('E:/AllDateSets/VerificationCodeDataSet/subimg/train_data', 40, mode='test')

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# 因为验证和测试在每一轮都对所有验证和测试样本都做验证和测试，所以无需 shuffle
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_db, batch_size=BATCH_SIZE, num_workers=2)

# 使用可视化工具
viz = visdom.Visdom()


# 定义一个用来计算正确率的函数
def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)     # 这一批数据的总数据量
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():   # 表示固定网络参数，即不需要 backforward，表现为下面的输入 x 不会对 model 的参数造成改变
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()   # item() 将 tensor 类型转成实数类型
    return correct / total


def evaluate_test(model, loader):
    """用来计算正确率的函数，但是还会将每一批测试结果通过 visdom 展示出来，并比较愿标签和预测标签"""
    import time
    import visdom
    viz_in = visdom.Visdom()
    correct = 0
    total = len(loader.dataset)  # 这一批数据的总数据量
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():  # 表示固定网络参数，即不需要 backforward，表现为下面的输入 x 不会对 model 的参数造成改变
            # nrow = 8 表示每行显示 8 个

            # 逆正则化的过程，主要是保证图片再显示的时候不会出现颜色上的问题
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)  # unsqueeze(1) 表示在后面插入一个维度
            std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
            x = x.cpu() * std + mean

            viz_in.images(x, nrow=8, win='batch', opts=dict(title='batch'))     # 显示图片
            viz_in.text(str(y.cpu().numpy()), win='label', opts=dict(title='batch-y'))      # 显示对应的标签

            logits = model(x.cuda())
            pred = logits.argmax(dim=1)

            print(torch.eq(pred, y).sum().float().item() / len(x))  # 这一批数据（32）的准确率

            viz_in.text(str(pred.cpu().numpy()), win='pred', opts=dict(title='batch-pred'))     # 显示预测标签
            time.sleep(5)

        # 总的准确率
        correct += torch.eq(pred, y).sum().float().item()  # item() 将 tensor 类型转成实数类型
    return correct / total


def training():
    """
    训练模型
    :return:
    """
    # 共 62 个类别
    model = ResNet18(62).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criteon_loss_fun = nn.CrossEntropyLoss()        # 损失函数记得加括号，否则出错

    best_acc, best_epoch = 0, 0     # 记录最好的正确率和训练轮次
    global_step = 0     # 全局的 step 计数，用于计数 visdom 中每步更新状态的 x 轴坐标
    # 使用 visdom 工具将每次的状态都保存下来
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 40, 40], y: [b]（注意这里的 y 共 b 行，每行一个数字代表一个分类）
            x, y = x.to(device), y.to(device)

            logits = model(x)
            # 在 pytorch 中的 loss 计算中，会自动的在内部实现 onehot 编码，所以这里只需传入 y（[b]）即可
            loss = criteon_loss_fun(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 在这里记录和更新 loss 曲线
            # y 轴的值就是 loss 的值
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        # 每训练两个轮次做一次测试
        if epoch % 2 == 0:
            val_acc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = EPOCHS
                # 将当前训练结果最好的模型状态保存下来
                torch.save(model.state_dict(), 'best_ver.mdl')

                # 在这里记录和更新 val_acc 曲线（这里并不更新global_step，即每训练两轮才更新一次 val_acc 曲线，迭代速度较慢）
                viz.line([val_acc], [global_step], win='val_acc', update='append')
                # 打印当前 loss 和 acc
            print('epoch:', epoch, 'loss:', loss.item(), 'val_acc:', val_acc)

    print('best_acc:', best_acc, 'best_epoch', best_epoch)
    print('------------------------ training is end -------------------------')


def validating():
    """
    使用测试集测试模型效果
    :return:
    """
    # 将训练过程得到的最好的结果模型加载进来，用于后面的 test 集测试
    model = ResNet18(62).to(device)
    model.load_state_dict(torch.load('best_ver.mdl'))
    print('loaded from ckpt!')

    # 下面使用测试集进行测试
    test_acc = evaluate_test(model, test_loader)

    print('test_acc:', test_acc)
    print('--------------------- validating is end -----------------------')


if __name__ == '__main__':
    # training()
    validating()
