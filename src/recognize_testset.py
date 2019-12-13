"""
利用训练好的模型对测试集中验证码进行识别
"""
import torch
from PIL import Image
from matplotlib.pylab import plt
import os
from torchvision import transforms
from resnet import ResNet18
import fileoperation as fp
import img_operations as iop
import csvoperation as cop

device = torch.device('cuda')

# 类别对应字典
CLASS = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
         '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E', '15': 'F', '16': 'G', '17': 'H', '18': 'I', '19': 'J', '20': 'K', '21': 'L', '22': 'M', '23': 'N', '24': 'O', '25': 'P', '26': 'Q', '27': 'R', '28': 'S', '29': 'T', '30': 'U', '31': 'V', '32': 'W', '33': 'X', '34': 'Y', '35': 'Z',
         '36': 'a', '37': 'b', '38': 'c', '39': 'd', '40': 'e', '41': 'f', '42': 'g', '43': 'h', '44': 'i', '45': 'j', '46': 'k', '47': 'l', '48': 'm', '49': 'n', '50': 'o', '51': 'p', '52': 'q', '53': 'r', '54': 's', '55': 't', '56': 'u', '57': 'v', '58': 'w', '59': 'x', '60': 'y', '61': 'z'}


def prediction_for_all(file_path, save_path):
    """
    对所有图片进行预测，并将结果存入 csv 文件中
    :param save_path: csv 文件要保存的路径
    :param file_path: 测试图片存放路径
    :return:
    """
    # 1.首先得到测试文件列表（当前路径下的所有图片文件）
    img_list = fp.get_all_file_list(file_path)
    print(img_list)

    # 2.加载模型
    model = ResNet18(62).to(device)
    model.load_state_dict(torch.load('best_ver.mdl'))
    print('loading model success!')

    # 3.对每张图片进行预测
    dicts = {'ID': 'label'}
    for img in img_list:
        print(img)
        result = prediction_for_img(img, model)
        print(result)
        # print(os.path.split(img)[-1])
        dicts[os.path.split(img)[-1]] = result

    print(dicts)
    # 4.将结果存入 csv 文件
    cop.save_to_csv(save_path, 'submission_test.csv', dicts)


def prediction_for_img(img_path, model):
    """
    对一张图片进行预测
    :param model: 训练好的模型
    :param img_path: 图片存放路径
    :return: 返回图片预测结果的字符串
    """
    # 1.读入
    img = Image.open(img_path)
    # 2.分割成单字符图片
    region1, region2, region3, region4 = iop.subimg(img)
    # 3.对单字符图片进行数据增强，该过程传入的是原始的 Image 类型的数据，而不是 numpy 数组类型的数据
    tf = transforms.Compose([
        transforms.Resize((int(40 * 1.25), int(40 * 1.25))),
        transforms.ToTensor(),  # 转成 tensor 数据类型
        # 归一化（这一步不可或缺，因为训练的过程对所有数据都使用了归一化过程，所以这里使用 model 时需要保证输入经过了一样的处理）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    region1, region2, region3, region4 = tf(region1), tf(region2), tf(region3), tf(region4)

    # 为了加快处理速度，将加载模型步骤提到外面，避免了每次都要重新加载模型的时间耗费

    # 5.进行预测
    # 5.1.在第 0 个位置插入一个维度使得 [3, 50, 50] => [1, 3, 50, 50]
    region1, region2, region3, region4 = region1.unsqueeze(0), region2.unsqueeze(0), region3.unsqueeze(0), region4.unsqueeze(0)
    # 5.2.对每个单独的字符图片进行预测
    with torch.no_grad():
        logit1, logit2, logit3, logit4 = model(region1.to(device)), model(region2.to(device)), model(region3.to(device)), model(region4.to(device))
    # 5.3.选出概率最大值所在的位置（也即分类）
    pred1, pred2, pred3, pred4 = logit1.argmax(dim=1), logit2.argmax(dim=1), logit3.argmax(dim=1), logit4.argmax(dim=1)
    # 5.4.拼接得到整张图片的预测字符串
    verification_ = CLASS[str(pred1.item())] + CLASS[str(pred2.item())] + CLASS[str(pred3.item())] + CLASS[str(pred4.item())]

    # 6.返回预测结果
    return verification_


def test():
    """对一张图片单独做测试，并展示效果"""
    img_path = 'E:/AllDateSets/VerificationCodeDataSet/test/2.jpg'
    img = Image.open(img_path)
    region1, region2, region3, region4 = iop.subimg(img)

    # 加载图片到 matplotlib 中
    plt.subplot(151), plt.imshow(img)
    plt.subplot(152), plt.imshow(region1)
    plt.subplot(153), plt.imshow(region2)
    plt.subplot(154), plt.imshow(region3)
    plt.subplot(155), plt.imshow(region4)

    # 数据增强
    # 该过程传入的是原始的 Image 类型的数据，而不是 numpy 数组类型的数据
    tf = transforms.Compose([
        transforms.Resize((int(40 * 1.25), int(40 * 1.25))),
        # 转成 tensor 数据类型
        transforms.ToTensor(),
        # 归一化（这一步不可或缺，因为训练的过程对所有数据都使用了归一化过程，所以）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    region1, region2, region3, region4 = tf(region1), tf(region2), tf(region3), tf(region4)
    # print(region1.shape)    # torch.Size([3, 50, 50])

    model = ResNet18(62).to(device)
    model.load_state_dict(torch.load('best_ver.mdl'))
    print('loading model success!')

    # 在第 0 个位置插入一个维度使得 [3, 50, 50] => [1, 3, 50, 50]
    region1, region2, region3, region4 = region1.unsqueeze(0), region2.unsqueeze(0), region3.unsqueeze(0), region4.unsqueeze(0)
    print(region1.shape)    # torch.Size([1, 3, 50, 50])
    with torch.no_grad():
        logit1 = model(region1.to(device))
        logit2 = model(region2.to(device))
        logit3 = model(region3.to(device))
        logit4 = model(region4.to(device))
    pred1 = logit1.argmax(dim=1)  # 选出概率最大值所在的位置（也即分类）
    pred2 = logit2.argmax(dim=1)
    pred3 = logit3.argmax(dim=1)
    pred4 = logit4.argmax(dim=1)
    print("预测结果 :", CLASS[str(pred1.item())], CLASS[str(pred2.item())], CLASS[str(pred3.item())], CLASS[str(pred4.item())])

    verification_ = CLASS[str(pred1.item())] + CLASS[str(pred2.item())] + CLASS[str(pred3.item())] + CLASS[str(pred4.item())]
    # 显示
    plt.text(-65, 60, "Prediction results: " + verification_)   # 预测结果
    plt.show()


if __name__ == '__main__':
    # main()
    # test()
    file_path = 'E:/AllDateSets/VerificationCodeDataSet/test'
    save_path = 'E:/AllDateSets/VerificationCodeDataSet/'
    prediction_for_all(file_path, save_path)
