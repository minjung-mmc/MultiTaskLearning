import os
import glob
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_math_ops import imag
import time
import labels
import torch
import numpy as np
import config
from torchvision.transforms import ToPILImage
from effdet import generate_detections
import matplotlib.patches as patches
import random

# import tensorflow as tf


def get_names(path, purpose):
    if purpose == "image":
        output = sorted(glob.glob(path + "/**/*.png", recursive=True))
    elif purpose == "label":
        output = sorted(glob.glob(path + "/**/*labelIds.png", recursive=True))
    return output


def number_names(names):
    numbered_names = {}
    for idx, name in enumerate(names):
        numbered_names[idx] = name
    return numbered_names


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


def visualize(path, save_path, bbox, target, epoch=0, idx=0):
    fig, ax = plt.subplots(1, len(path), figsize=[30, 5])
    color = get_cmap(30)

    # print(path[0][0].shape)
    for i in range(len(path)):
        ax[i].imshow(path[i])
    if type(bbox) != int:
        for i in range(len(bbox)):
            n = random.randrange(1, 30)
            # print("bbox: ", bbox[i][0].item(), bbox[i][1].item(), bbox[i][2].item(), bbox[i][3].item())
            [x1, y1, x2, y2] = bbox[i][:4]
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1,
                edgecolor=color(n),
                facecolor="none",
            )
            ax[-2].add_patch(rect)
            id = int(bbox[i][-1].item())
            text = labels.id2label[id - 1].name
            ax[-2].text(x1, y1, "{}".format(text), color=color(n))
    box_tar = target["bbox"][0]
    cls = target["cls"][0]

    ############################# Filter -1 ###############################
    valid_idx = cls > -1  # filter gt targets w/ label <= -1
    cls = cls[valid_idx]
    box_tar = box_tar[valid_idx]
    ############################# Filter -1 ###############################

    # print(box_tar.shape)
    for i in range(len(box_tar)):
        n = random.randrange(1, 30)
        [y1, x1, y2, x2] = box_tar[i][:4] / 8
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor=color(n),
            facecolor="none",
        )
        ax[-1].add_patch(rect)
        id = cls[i].item()
        text = labels.id2label[id - 1].name
        ax[-1].text(x1, y1, "{}".format(text), color=color(n))

    plt.savefig(save_path + "/img_{}_{}.png".format(epoch, idx), bbox_inches='tight', pad_inches=0)
    plt.close()


def get_path():
    folder = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))
    path_weight = "./weights/" + folder
    path_img = "./result/" + folder

    make_dir(path_weight)
    make_dir(path_img)
    return path_weight, path_img


def make_dir(path):
    if os.path.exists(path):
        # print("There already exists folder")
        pass
    else:
        os.mkdir(path)


def decode_segmap(seg):
    """
        input: seg = size([1024,2048])
        output: image of seg map  
    """
    color_tensor = torch.ByteTensor(3, seg.size(0), seg.size(1)).to(
        config.params.device
    )
    for x in labels.id2color.items():
        mask = torch.eq(seg, x[0])  # size = (1024, 2048)
        for channel, color_value in enumerate(x[1]):
            color_tensor[channel].masked_fill_(
                mask, color_value
            )  # size = (3, 1024, 2048)

    return color_tensor


def save_network(net, path, epoch):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    torch.save(net.state_dict(), path + "/checkpoint_epoch{}.pth".format(epoch))


def get_ann_names(img_name):
    ann = img_name.split("/")[-1].split("_")[:3]
    ann = "_".join(ann) + "_gtFine_panoptic.png"
    return ann


def post_process(
    cls_outputs, box_outputs, num_levels, num_classes, max_detection_points=5000
):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].

        num_levels (int): number of feature levels

        num_classes (int): number of output classes
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat(
        [
            cls_outputs[level]
            .permute(0, 2, 3, 1)
            .reshape([batch_size, -1, num_classes])  # size = (batch, )
            for level in range(num_levels)
        ],
        1,
    )
    # print("after: ", cls_outputs_all.size())  # torch.Size([128, 110484, 90])

    box_outputs_all = torch.cat(
        [
            box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
            for level in range(num_levels)
        ],
        1,
    )

    # print("after: ", box_outputs_all.size())  # torch.Size([128, 110484, 4])

    _, cls_topk_indices_all = torch.topk(
        cls_outputs_all.reshape(batch_size, -1), dim=1, k=max_detection_points
    )

    # print("cls_outputs_all: ", cls_outputs_all.reshape(batch_size, -1).size())
    # print("cls_topk_indices_all: ", cls_topk_indices_all.size())

    indices_all = cls_topk_indices_all // num_classes
    classes_all = cls_topk_indices_all % num_classes

    # print("indices_all:  ", indices_all.shape)  # torch.Size([128, 5000])
    # print("classes_all:  ", classes_all.shape)

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4)
    )

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, num_classes)
    )
    # print(
    #     "cls_outputs_all_after_topk : (before) ", cls_outputs_all_after_topk.shape
    # )  # torch.Size([128, 5000, 90])
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2)
    )
    # print(
    #     "cls_outputs_all_after_topk : (after) ", cls_outputs_all_after_topk.shape
    # )  # torch.Size([128, 5000, 1])
    # print(
    #     "box_outputs_all_after_topk : (after) ", box_outputs_all_after_topk.shape
    # )  # torch.Size([128, 5000, 4])

    return (
        cls_outputs_all_after_topk,
        box_outputs_all_after_topk,
        indices_all,
        classes_all,
    )


# @torch.jit.script
def batch_detection(
    batch_size,
    class_out,
    box_out,
    anchor_boxes,
    indices,
    classes,
    img_scale=None,
    img_size=None,
    max_det_per_image=100,
    soft_nms=False,
):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        img_scale_i = None if img_scale is None else img_scale[i]
        img_size_i = None if img_size is None else img_size[i]
        detections = generate_detections(
            class_out[i],
            box_out[i],
            anchor_boxes,
            indices[i],
            classes[i],
            img_scale_i,
            img_size_i,
            max_det_per_image=max_det_per_image,
            soft_nms=soft_nms,
        )
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)


def decode_det(res, nms_conf=0.3, obj_thresh=0.5):
    write = False
    for ind in range(1):
        image = res[ind]  # 한장의 이미지 [100,6]

        # obj thresh Mask
        image = image[image[:, -2] > obj_thresh]
        # print("after: ", image.shape)

        img_classes = unique(image[:, -1])
        # score_mask = image[]
        for cls in img_classes:
            ############################ 해당 클래스만 mask #################################
            image_pred = image[image[:, -1] == cls]
            ############################ Objectness 순서대로 sort ###############################
            conf_sort_index = torch.sort(image_pred[:, 4], descending=True)[
                1
            ]  # object score가 큰 순서대로 정렬 / 1 은 indices
            image_pred = image_pred[conf_sort_index]
            idx = image_pred.size(0)  # class에 해당하는 box 수
            ###################### 해당 class 의 박스마다 nms 실행 ####################################
            for i in range(idx):
                try:
                    ious = bbox_iou(
                        image_pred[i].unsqueeze(0), image_pred[i + 1 :]
                    )  # row
                except ValueError:  # slice 할 때 empty tensor가 있을 때
                    break
                except IndexError:  # ind를 없애 Out of bounds 될 때
                    break
                iou_mask = (
                    (ious < nms_conf).float().unsqueeze(1)
                )  # iou가 threshold보다 작으면 1
                image_pred[i + 1 :] *= iou_mask  # objectness가 더 작은 박스 중에서 삭제

                non_zero_ind = torch.nonzero(image_pred[:, 4]).squeeze()
                image_pred = image_pred[non_zero_ind].view(-1, 6)

            if not write:
                output = image_pred
                write = True
            else:
                out = image_pred
                output = torch.cat((output, out))
    try:
        return (
            output
        )  # row-True Detection col-batch, x, y, w, h, o, maxConf, maxConfInd []
    except:
        return 0  # 배치 안에 이미지 중 detect된 것이 하나도 없었음.


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding box
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle (Intersection Coordinate 계산)
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou  # size (:, 1)


def unique(tensor):
    tensor_np = tensor.detach().cpu().numpy()  # tensor to numpy
    unique_np = np.unique(tensor_np)  # numpy unique classes.
    unique_tensor = torch.from_numpy(unique_np)  # tensor unique classes.

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


# print(labels.id2color[5])

