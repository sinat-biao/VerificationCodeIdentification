"""
CSV 文件操作
"""
import csv
import os


def save_to_csv(store_path, csv_name, data_list):
    """
    将数据存入 csv 文件
    :param data_list: 数据列表
    :param csv_name: csv 文件的名字
    :param store_path: csv 文件保存路径
    :return:
    """
    # 如果存在同名文件，就删除
    if os.path.exists(os.path.join(store_path, csv_name)):
        print('csv file is existed, now it will be remove.')
        os.remove(os.path.join(store_path, csv_name))
    # 存入数据
    with open(os.path.join(store_path, csv_name), mode='w', newline='') as f:
        writer = csv.writer(f)      # csv 形式比较简单，就是每一行用逗号分开
        for k, v in data_list.items():
            writer.writerow([k, v])
    print('wirten into csv file:', csv_name)


def read_csv(path):
    """
    读取 csv 文件，并将每一行拆分成 id 和 label，组成 ID 和 label 数组返回
    :param path: csv 文件存储路径
    :return: id 和 对应的 label 数组
    """
    csv_file = csv.reader(open(path))
    # print(csv_file)
    # 获取每一行：
    id = []
    label = []
    for item in csv_file:
        if not len(item) == 0:
            # print(item[0], item[1])
            id.append(item[0])
            label.append(item[1])
    # 第一行是 title，不需要，故只返回 [1:]
    return id[1:], label[1:]


if __name__ == '__main__':
    path = "E:/VerificationCodeDataSet/train/train_label.csv"
    read_csv(path)
