import numpy as np

import torch
import matplotlib.pyplot as plt

import collections
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map):
    boxess = torch.from_numpy(boxes)
    scoress = torch.from_numpy(scores)
    classess = torch.from_numpy(classes)
    # Remove the overlapped boxes
    keep = nms(boxess, scoress, 0.1)
    boxes_s, scores_s, classes_s = boxess[keep], scoress[keep], classess[keep]
    # Remove the low-scoring boxes
    for i in range(boxes_s.shape[0]):
        if scores_s[i] > thresh:
            box = tuple(boxes_s[i].tolist())  # numpy -> list -> tuple
            a = int(classes_s[i])
            if a in category_index.keys():
                class_name = category_index[a]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            display_str = '{}: {}%'.format(display_str, int(100 * scores_s[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = STANDARD_COLORS[
                a % len(STANDARD_COLORS)]
        else:
            break


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def closest(mylist, number):
    answer = []
    for i in mylist:
        answer.append(abs(number - i))
    return answer.index(min(answer))


def draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color):
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the string added to the top of the bounding box exceeds
    # the height of the image, the string is added below the bounding box
    display_str_heights = [font.getsize(ds)[1] for ds in box_to_display_str_map[box]]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # reverse list and print from bottom to top.
    for display_str in box_to_display_str_map[box][::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_box(filename, image, eic, boxes, classes, scores, category_index, thresh, line_thickness=8):
    x_min = []
    x_max = []
    mz_min = []
    mz_max = []
    mz_mean = []
    total_intensity = []
    eic_name = []
    peak_height = []
    rt_time = []
    rt_duration = []
    p_gao = []
    time = eic.rt
    yy = eic.i[2:-2]
    mmzz = eic.mz
    xx = np.linspace(101, 554, len(yy))
    rt = np.linspace(time[0], time[1], len(yy))
    h = rt[1] - rt[0]

    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

    filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map)

    # draw all boxes onto an image.
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        start_index = closest(xx, xmin)
        end_index = closest(xx, xmax)
        point_numbers = end_index - start_index + 1
        intensity = yy[start_index:end_index + 1]
        height = max(intensity)
        gao = height - min(intensity)
        no_zero_numbers = sum(i > 0 for i in intensity)
        # set min_peak_width and intensity threshold at peak apex
        if no_zero_numbers >= 5 and height > 300:
            mz = mmzz[start_index:end_index + 1]
            sum_intensity = 2 * sum(intensity) - intensity[0] - intensity[-1]
            totalintensity = (sum_intensity * h) / 2
            mzmean = np.mean(mz)
            mzmin = min(mz)
            mzmax = max(mz)
            rt_dur = rt[end_index] - rt[start_index]
            a = intensity.index(max(intensity)) - 2
            b = intensity.index(max(intensity)) + 2
            c = intensity.index(max(intensity)) + start_index
            if a >= 0 and b <= point_numbers - 1:
                x_min.append(rt[start_index])
                x_max.append(rt[end_index])
                rt_time.append(rt[c])
                rt_duration.append(rt_dur)
                p_gao.append(gao)
                mz_min.append(mzmin)
                mz_max.append(mzmax)
                mz_mean.append(mzmean)
                peak_height.append(height)
                total_intensity.append(totalintensity)
                eic_name.append(filename)
                (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                              ymin * 1, ymax * 1)
                draw.line([(left, top), (left, bottom), (right, bottom),
                           (right, top), (left, top)], width=line_thickness, fill=color)
                draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)
                plt.imshow(image)
                # saves the predicted bbox results to the specified file
                image.save("your path/result/result_" + filename)
                plt.clf()
    return eic_name, total_intensity, mz_mean, mz_min, mz_max, x_min, x_max, peak_height, rt_time, rt_duration, p_gao