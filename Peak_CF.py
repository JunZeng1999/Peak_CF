import time
import json
import numpy as np

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_box
from backbone import Classifier

from get_data import get_EICs
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # read the data file and get the EIC images
    parent = os.path.realpath('example.mzML')
    file = os.path.basename(parent)
    eics = get_EICs(file, delta_mz=0.005, required_points=15, dropped_points=3, progress_callback=None)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # pretreatment method
    data_transform1 = transforms.Compose(
        [transforms.CenterCrop((354, 472)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # create model
    model_class = Classifier(num_classes=2).to(device)
    model = create_model(num_classes=4)

    # load weights
    weights_path = "./save_weights/Classifier.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model_class.load_state_dict(torch.load(weights_path, map_location=device))
    model_class.to(device)

    train_weights = "./save_weights/resNetFpn-model.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    json_path1 = './class_indices.json'
    assert os.path.exists(json_path1), "file: '{}' dose not exist.".format(json_path1)
    json_file1 = open(json_path1, "r")
    class_indict = json.load(json_file1)

    label_json_path = './peak_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # data process
    aa = 0
    ttime = []
    eic_name = []
    total_intensity = []
    mz_mean = []
    mz_min = []
    mz_max = []
    rt_min = []
    rt_max = []
    peak_height = []
    peak_qiang = []
    rt_time = []
    rt_duration = []
    eic_name2 = []
    total_intensity2 = []
    mz_mean2 = []
    mz_min2 = []
    mz_max2 = []
    rt_min2 = []
    rt_max2 = []
    p_gao2 = []
    peak_height2 = []
    rt_time2 = []
    rt_duration2 = []
    for eic in eics:
        x = eic.rt
        y = eic.i[2:-2]
        x = np.linspace(x[0], x[1], len(y))
        # noise intensity
        threshold1 = 20
        if max(y) > threshold1:
            plt.plot(x, y)
            plt.xlabel('Retention time (minutes)')
            plt.ylabel('Intensity')
            plt.title('mz = %.4f' % eic.mzmean)
            # path for saving EIC images
            plt.savefig("your path/photo/your_image_name{:06}.jpg".format(aa))
            plt.clf()
            # load images
            filename = "your_image_name{:06}.jpg".format(aa)
            dir = "your path/photo/"
            newdir = os.path.join(dir, filename)
            original_img = Image.open(newdir)
            signal = original_img

            # from pil image to tensor, do not normalize image
            img_class = data_transform1(original_img)
            img_class = torch.unsqueeze(img_class, dim=0)

            # prediction
            model_class.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model_class(img_class.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            if predict_cla != 0:
                plt.imshow(signal)
                # save the image results of the classification prediction
                signal.save("your path/peak/s_" + filename)
                plt.clf()
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(original_img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)

                model.eval()
                with torch.no_grad():
                    # init
                    img_height, img_width = img.shape[-2:]
                    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                    model(init_img)

                    t_start = time_synchronized()
                    predictions = model(img.to(device))[0]
                    t_end = time_synchronized()
                    times = t_end - t_start
                    ttime.append(times)

                    predict_boxes = predictions["boxes"].to("cpu").numpy()
                    predict_classes = predictions["labels"].to("cpu").numpy()
                    predict_scores = predictions["scores"].to("cpu").numpy()

                    if len(predict_boxes) == 0:
                        print("No target was detected!")

                    eic_name1, total_intensity1, mz_mean1, mz_min1, mz_max1, rt_min1, \
                    rt_max1, peak_height1, rt_time1, rt_duration1, p_gao1 = draw_box(filename,
                                                                                     original_img,
                                                                                     eic,
                                                                                     predict_boxes,
                                                                                     predict_classes,
                                                                                     predict_scores,
                                                                                     category_index,
                                                                                     thresh=0.5,
                                                                                     line_thickness=1)

                    eic_name2 += eic_name1
                    total_intensity2 += total_intensity1
                    mz_mean2 += mz_mean1
                    mz_min2 += mz_min1
                    mz_max2 += mz_max1
                    rt_min2 += rt_min1
                    rt_max2 += rt_max1
                    p_gao2 += p_gao1
                    peak_height2 += peak_height1
                    rt_time2 += rt_time1
                    rt_duration2 += rt_duration1
            aa += 1
    order_mat = np.array(mz_mean2)
    order = order_mat.argsort()
    for i in range(0, len(order)):
        eic_name.append(eic_name2[order[i]])
        total_intensity.append(total_intensity2[order[i]])
        mz_min.append(mz_min2[order[i]])
        mz_mean.append(mz_mean2[order[i]])
        mz_max.append(mz_max2[order[i]])
        rt_min.append(rt_min2[order[i]])
        rt_time.append(rt_time2[order[i]])
        rt_max.append(rt_max2[order[i]])
        peak_qiang.append(p_gao2[order[i]])
        peak_height.append(peak_height2[order[i]])
        rt_duration.append(rt_duration2[order[i]])

    df = pd.DataFrame()
    df['row m/z'] = mz_mean
    df['row retention time'] = rt_time
    df['Eic_name'] = eic_name
    df['Peak m/z'] = mz_mean
    df['Peak RT'] = rt_time
    df['Peak RT start'] = rt_min
    df['Peak RT end'] = rt_max
    df['Peak duration time'] = rt_duration
    df['Peak height'] = peak_height
    df['Peak height difference'] = peak_qiang
    df['Peak area'] = total_intensity
    df['Peak m/z min'] = mz_min
    df['Peak m/z max'] = mz_max
    # path for saving the detected result
    df.to_csv("your path/result/your_result_name.csv")
    t = np.sum(ttime)
    print("inference time: {}".format(t))


if __name__ == '__main__':
    main()
    
