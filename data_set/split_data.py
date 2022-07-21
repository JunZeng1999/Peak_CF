import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    random.seed(0)

    # validation sets
    split_rate = 0.36

    # Eic_photo folder
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "Eic_data")
    origin_Eic_path = os.path.join(data_root, "Eic_photo")
    assert os.path.exists(origin_Eic_path), "path '{}' does not exist.".format(origin_Eic_path)

    Eic_class = [cla for cla in os.listdir(origin_Eic_path)
                 if os.path.isdir(os.path.join(origin_Eic_path, cla))]

    # Create a folder to save the training set
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in Eic_class:
        # Create folders for each category
        mk_file(os.path.join(train_root, cla))

    # Create a folder to save validation sets
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in Eic_class:
        # Create folders for each category
        mk_file(os.path.join(val_root, cla))
    
    # Create a folder to save test sets
    test_root = os.path.join(data_root, "test")
    mk_file(test_root)
    for cla in Eic_class:
        # Create folders for each category
        mk_file(os.path.join(test_root, cla))
    a = 0.75

    for cla in Eic_class:
        cla_path = os.path.join(origin_Eic_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # random sampling
        eval_index = random.sample(images, k=int(num*split_rate))
        num1 = int(len(eval_index) * a)
        eval_index1 = eval_index[0:num1]
        eval_index2 = eval_index[num1:]
        for index, image in enumerate(images):
            if image in eval_index1:
                # copy the files assigned to the validation set to the appropriate directory
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            elif image in eval_index2:
                # copy the files assigned to the test set to the appropriate directory
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(test_root, cla)
                copy(image_path, new_path)
            else:
                # copy the files assigned to the training set to the appropriate directory
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)

    print("processing done!")


if __name__ == '__main__':
    main()
