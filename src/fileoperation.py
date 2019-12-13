"""
操作文件
"""
import os


def get_all_file_list(root_path):
    """
    返回当前目录下的所有文件列表
    :return:
    """
    file_list = []
    # key=lambda x: int(x[:-4]) : 倒着数第四位'.'为分界线，按照'.'左边的数字从小到大排序
    for f in sorted(os.listdir(os.path.join(root_path)), key=lambda x: int(x[:-4])):   # 获取的列表是乱序的，记得排一下序
        sub_path = os.path.join(root_path, f)
        if os.path.isfile(sub_path) and sub_path.endswith('jpg'):
            file_list.append(sub_path)
    return file_list


def delete_all_file(root_path):
    """
    递归删除 root_path 目录下所有的文件夹中的文件（只删文件，不删文件夹）
    :param root_path:
    :return:
    """
    # os.listdir(path_data) 返回一个列表，里面是当前目录下面的所有的文件和文件夹名
    for f in os.listdir(root_path):
        sub_path = root_path + f
        if os.path.isfile(sub_path):    # sub_path 指向的是文件，就删除
            print("find file:", sub_path)
            os.remove(sub_path)
        else:                           # 否则就递归删除操作
            delete_all_file(sub_path+"/")


if __name__ == '__main__':
    # r_path = "E:/AllDateSets/VerificationCodeDataSet/subimg/train_data/"
    r_path = "E:/AllDateSets/VerificationCodeDataSet/subimg/test_data/"
    delete_all_file(r_path)
