"""
This script is used to create a training set/validation set
"""


import os
import random


def main():
    random.seed(0)
import os
import random


def main():
    random.seed(0)

    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    val_num = len(val_index)/2
    train_files = []
    val_files = []
    test_files = []
    num = 0
    for index, file_name in enumerate(files_name):
        if index in val_index:
            num = num + 1
            if num < val_num:
                val_files.append(file_name)
            else:
                test_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        test_f = open("test.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
        test_f.write("\n".join(test_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()

    files_path = "./VOCdevkit/VOC2012/Annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.5

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("train.txt", "x")
        eval_f = open("val.txt", "x")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
