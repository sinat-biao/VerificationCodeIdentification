from PIL import Image
from matplotlib.pylab import plt
import numpy as np
import csvoperation


# 截取图像
def subimg(image):
    box1 = (10, 0, 35, 40)   # 左，上，右，下
    box2 = (35, 0, 60, 40)
    box3 = (60, 0, 85, 40)
    box4 = (85, 0, 110, 40)

    region1 = image.crop(box1)
    region2 = image.crop(box2)
    region3 = image.crop(box3)
    region4 = image.crop(box4)
    return region1, region2, region3, region4


# 去除干扰线
# 对区域求平均，看是否能消除干扰线
# 对 3*3 区域求平均（相当于进行平均池化）
def averge_of_nine_pix(img_arr):
    height, width = img_arr.shape
    for i in range(0, height):
        for j in range(0, width):
            if i == 0 or j == 0 or i == height-1 or j == width-1:
                pass
            else:
                sum = 0
                sum += img_arr[i][j]
                sum += img_arr[i-1][j-1]
                sum += img_arr[i-1][j]
                sum += img_arr[i-1][j+1]
                sum += img_arr[i][j-1]
                sum += img_arr[i][j+1]
                sum += img_arr[i+1][j-1]
                sum += img_arr[i+1][j]
                sum += img_arr[i+1][j+1]
                avg = sum / 9
                img_arr[i][j] = avg
    return img_arr


# 去除干扰线
# 对区域求平均，看是否能消除干扰线
# 对十字区域求平均
def average_shizi(img_arr):
    height, width = img_arr.shape
    for i in range(2, height-2):
        for j in range(2, width-2):
            sum = 0
            sum = sum + img_arr[i][j] + img_arr[i - 2][j] + img_arr[i - 1][j] + img_arr[i+1][j] + img_arr[i+2][j]
            sum = sum + img_arr[i][j - 1] + img_arr[i][j + 1]
            avg = sum / 7
            img_arr[i][j] = avg
    return img_arr


def isnumber(str):
    if "0123456789".__contains__(str):
        return True
    else:
        return False


def islowercaseletter(str):
    if "abcdefghijklmnopqrstuvwxyz".__contains__(str):
        return True
    else:
        return False


def iscapitalletters(str):
    if "ABCDEFGHIJKLMNOPQRSTUVWXYZ".__contains__(str):
        return True
    else:
        return False


def initial_counts():
    """
    为 0-9 a-z A-Z 共 62 个类别分别计数，初始化计数 map
    :return:
    """
    counts = {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1,
              'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1, 'h': 1, 'i': 1, 'j': 1, 'k': 1, 'l': 1, 'm': 1, 'n': 1, 'o': 1, 'p': 1, 'q': 1, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 1, 'w': 1, 'x': 1, 'y': 1, 'z': 1,
              'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 1, 'W': 1, 'X': 1, 'Y': 1, 'Z': 1
              }
    return counts


def test():
    """test image cutting"""
    image = Image.open("4.jpg")
    img_arr = np.array(image.convert('L'))
    np.savetxt("img_arr.txt", img_arr, fmt='%d')    # 保存 numpy 数组
    region1, region2, region3, region4 = subimg(image)      # 切分图片

    # 两种方式去除干扰线，肉眼效果均不是很好，等到后面使用神经网络训练再看看是否有用
    img_arr_9_pix = averge_of_nine_pix(img_arr)
    img_arr_shizi_pix = average_shizi(img_arr)
    # plt.subplot(231), plt.imshow(image)
    # plt.subplot(232), plt.imshow(region1)
    # plt.subplot(233), plt.imshow(region2)
    # plt.subplot(234), plt.imshow(region3)
    # plt.subplot(235), plt.imshow(region4)
    # plt.subplot(236), plt.imshow(img_arr, 'gray')
    plt.subplot(131), plt.imshow(image)
    plt.subplot(132), plt.imshow(img_arr_9_pix, 'gray')
    plt.subplot(133), plt.imshow(img_arr_shizi_pix, 'gray')
    plt.show()


def main():
    """
    处理原始图像数据集，将图像切分为单个字符图片进行保存（每个字符放在自己的单独的文件夹下保存，共 62 个文件夹）
    这里不对文件进行训练集和测试集的划分，后面在进行神经网络训练读取图片时会自动的进行划分
    :return:
    """
    # 1 - 首先读取 csv 文件，得到每张图片对应的验证码
    csv_train_path = "E:/AllDateSets/VerificationCodeDataSet/train/train_label.csv"
    id, label = csvoperation.read_csv(path=csv_train_path)
    # 2 - 按序读取图片，然后切分图片，并将切分后的图片保存到相应目录下
    # -> 这里直接将所有图片数据都切分然后存到一起，先不区分训练集和测试集，后面读取数据时自会区分
    # 初始化 counts，用来计数 0-9 a-z A-Z 的数量，用来命名保存的文件名
    counts = initial_counts()
    for i in range(len(id)):
        img_path = "E:/AllDateSets/VerificationCodeDataSet/train/" + id[i]
        image = Image.open(img_path)

        for (r, k) in zip(subimg(image), label[i][:]):
            if isnumber(k):
                savepath = 'E:/AllDateSets/VerificationCodeDataSet/subimg/train_data/' + k + '/' + str(counts[k]) + '.jpg'
                counts[k] += 1
            if islowercaseletter(k):
                savepath = 'E:/AllDateSets/VerificationCodeDataSet/subimg/train_data/l' + k + '/' + str(counts[k]) + '.jpg'
                counts[k] += 1
            if iscapitalletters(k):
                savepath = 'E:/AllDateSets/VerificationCodeDataSet/subimg/train_data/c' + k + '/' + str(counts[k]) + '.jpg'
                counts[k] += 1
            r.save(savepath)
        print(counts)


if __name__ == '__main__':
    # test()
    main()
