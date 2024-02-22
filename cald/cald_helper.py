import torchvision.transforms.functional as F
import torch.nn.functional as Fun
import random
import torch
import numpy as np
import PIL
from PIL import Image, ImageDraw
import datetime
import os
import time


def HorizontalFlipFeatures(image, features):
    image = F.to_tensor(image)
    image = image.flip(-1)
    new_features = {}
    for k in features:
        new_features[k] = features[k].detach().flip(-1)
    return image, new_features


def HorizontalFlip(image, bbox):
    if type(image) == Image.Image:
        image = F.to_tensor(image)
    height, width = image.shape[-2:]
    image = image.flip(-1)
    b = bbox.clone()
    b[:, [0, 2]] = width - bbox[:, [2, 0]]
    return image, b


def ResizeFeatures(img, features, ratio):
    if not type(img) == Image.Image:
        img = F.to_pil_image(img)
    w, h = img.size
    iw = int(w * ratio)
    ih = int(h * ratio)
    new_features = {}
    for k in features:
        fw = int(features[k].shape(-2) * ratio)
        fh = int(features[k].shape(-1) * ratio)
        new_features[k] = Fun.interpolate(features[k].detach(), (fw, fh))
    return F.to_tensor(img.Resize((iw, ih), Image.BILINEAR)), new_features


def Resize(img, boxes, ratio):
    if not type(img) == Image.Image:
        img = F.to_pil_image(img)
    w, h = img.resize
    ow = int(w * ratio)
    oh = int(h * ratio)
    return F.to_tensor(img.Resize((ow, oh), Image.BILINEAR)), boxes * ratio


def ColorSwap(image):
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms) - 1)]
    image = F.to_tensor(image)
    image = image[swap, :, :]
    return image


def ColorAdjust(image, factor):
    image = F.adjust_brightness(image, factor)
    image = F.adjust_contrast(image, factor)
    image = F.adjust_saturation(image, factor)
    return F.to_tensor(image)


def GaussianNoise(image, std=1):
    image = F.to_tensor(image)
    x = image + torch.randn(image.size()) * std / 255.0
    return x


def SaltPepperNoise(image, prob):
    image = F.to_tensor(image)
    noise = torch.rand(image.size())
    salt = torch.max(image)
    pepper = torch.min(image)
    image[noise < prob / 2] = salt
    image[noise > 1 - prob / 2] = pepper
    return image


def Cutout(
    image,
    boxes,
    labels,
    cut_num=2,
    fill_val=0,
    bbox_remove_thres=0.4,
    bbox_min_thres=0.1,
):
    if type(image) == Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    original_channel = image.size(0)

    count = 0
    for _ in range(50):
        cutout_size_h = random.uniform(0.05 * original_h, 0.2 * original_h)
        cutout_size_w = random.uniform(0.05 * original_w, 0.2 * original_w)

        left = random.uniform(0, original_w - cutout_size_w)
        right = left + cutout_size_w
        top = random.uniform(0, original_h - cutout_size_h)
        down = top + cutout_size_h
        cutout = torch.FloatTensor([int(left), int(top), int(right), int(down)])

        overlap_size = Intersect(cutout.unsqueeze(0), boxes)
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ratio = overlap_size / area_boxes
        if (
            ratio.max().item() > bbox_remove_thres
            or ratio.max().item() < bbox_min_thres
        ):
            continue

        cutout_arr = torch.full(
            (original_channel, int(down) - int(top), int(right) - int(left)), fill_val
        )
        image[:, int(top) : int(down), int(left) : int(right)] = cutout_arr
        count += 1
        if count >= cut_num:
            break

    return image


def Rotate(image, boxes, angle):
    if not type(image) == Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    new_boxes = boxes.clone()

    w = image.width
    h = image.height
    cx = w / 2
    cy = h / 2
    new_image = new_image.rotate(angle, expand=True)
    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)

    affine_matrix = torch.tensor(
        [
            [alpha, beta, (1 - alpha) * cx - beta * cy],
            [-beta, alpha, beta * cx + (1 - alpha) * cy],
        ]
    )

    box_width = (boxes[:, 2] - boxes[:, 0]).reshape(-1, 1)
    box_height = (boxes[:, 3] - boxes[:, 1]).reshape(-1, 1)

    x1 = boxes[:, 0].reshape(-1, 1)
    y1 = boxes[:, 1].reshape(-1, 1)

    x2 = x1 + box_width
    y2 = y1

    x3 = x1
    y3 = y1 + box_height

    x4 = boxes[:, 2].reshape(-1, 1)
    y4 = boxes[:, 3].reshape(-1, 1)

    corners = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)
    corners.reshape(len(boxes), 8)
    corners = corners.reshape(-1, 2)
    corners = torch.cat((corners, torch.ones(corners.shape[0], 1).cuda()), dim=1)

    cos = np.abs(affine_matrix[0, 0])
    sin = np.abs(affine_matrix[0, 1])

    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    affine_matrix[0, 2] += (nw / 2) - cx
    affine_matrix[1, 2] += (nh / 2) - cy

    rotate_corners = torch.mm(affine_matrix.cuda().float(), corners.t()).t()
    rotate_corners = rotate_corners.reshape(-1, 8)

    x_corners = rotate_corners[:, [0, 2, 4, 6]]
    y_corners = rotate_corners[:, [1, 3, 5, 7]]

    x_min, _ = torch.min(x_corners, dim=1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim=1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim=1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim=1)
    y_max = y_max.reshape(-1, 1)

    new_boxes = torch.cat((x_min, y_min, x_max, y_max))

    scale_x = new_image.width / w
    scale_y = new_image.height / h

    new_image = new_image.resize((w, h))

    new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y]).cuda()
    new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)

    return F.to_tensor(new_image), new_boxes


def Intersect(boxes1, boxes2):
    n1 = boxes1.size(0)
    n2 = boxes2.size(0)
    max_xy = torch.min(
        boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
        boxes2[:, 2:].unsqueeze(1).expand(n1, n2, 2),
    )

    min_xy = torch.max(
        boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
        boxes2[:, 2:].unsqueeze(1).expand(n1, n2, 2),
    )

    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]


import matplotlib.pyplot as plt


def draw_PIL_image(image, boxes, labels, scores, name):
    if not type(image) == Image.Image:
        image = F.to_pil_image(image)
    plt.imshow(image)
    plt.axis("off")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("vis/{}.png".format(name), dpi=256, bbox_inches="tight", pad_inches=0)
    plt.cla()


def draw_PIL_image_1(
    image, ref_boxes, boxes, ref_labels, labels, scores, pm, name, no=None
):
    if not type(image) == Image.Image:
        image = F.to_pil_image(image)
    plt.imshow(image)
    plt.axis("off")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    i = 0
    if no is not None:
        for n in no:
            if i < 1:
                color = "green"
                x, y = ref_boxes[n][0], ref_boxes[n][1]
                w, h = (
                    ref_boxes[n][2] - ref_boxes[n][0],
                    ref_boxes[n][3] - ref_boxes[n][1],
                )
                plt.text(
                    x,
                    y,
                    "{}={}".format(
                        voc_labels[ref_labels[n] - 1], round(scores[n].item(), 2)
                    ),
                    color="white",
                    verticalalignment="bottom",
                    bbox={"facecolor": color, "alpha": 0.1},
                    fontsize=24,
                )
            else:
                color = "red"
                x, y = boxes[n][0], boxes[n][1]
                w, h = boxes[n][2] - boxes[n][0], boxes[n][3] - boxes[n][1]
                plt.text(
                    x,
                    y,
                    "{}={}".format(voc_labels[labels[n] - 1], round(pm[n].item(), 2)),
                    color="white",
                    verticalalignment="bottom",
                    bbox={"facecolor": color, "alpha": 0.1},
                    fontsize=24,
                )
            i += 1
            plt.gca().add_patch(
                plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2.5)
            )
    plt.savefig("fig/{}".format(name), dpi=256, bbox_inches="tight", pad_inches=0)
    plt.cla()


def draw_PIL_image_2(image, boxes, ref_labels, name, no=None, color="green"):
    if not type(image) == Image.Image:
        image = F.to_pil_image(image)
    plt.imshow(image)
    plt.axis("off")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    i = 0
    plt.savefig("fig/o_{}.eps".format(name), dpi=256, bbox_inches="tight", pad_inches=0)
    plt.cla()


voc_labels = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

label_map = {k: v + 1 for v, k in enumerate(voc_labels)}

rev_label_map = {v: k for k, v in label_map.items()}

CLASSES = 20

distinct_colors = [
    "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
    for i in range(CLASSES)
]

label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

"""
@article{yu2021consistency,
  title={Consistency-based active learning for object detection},
  author={Yu, Weiping and Zhu, Sijie and Yang, Taojiannan and Chen, Chen and Liu, Mengyuan},
  journal={arXiv preprint arXiv:2103.10374},
  year={2021}
}
"""
