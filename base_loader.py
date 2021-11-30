import os
from re import M
import numpy as np

from torch.utils import data
from config import params
import cv2
from utils import *
import torch
import torchvision.transforms as transforms
import json


class BaseLoader(data.Dataset):
    def __init__(self, mode):
        super(BaseLoader, self).__init__()
        # self.ids = param.
        if mode == "train":
            img_path = params.train_img_path
            seg_label = params.train_seg_label
            depth_label = params.train_depth_label
            with open(params.train_ann) as json_file:
                self.ann = json.load(json_file)["annotations"]
        elif mode == "val":
            img_path = params.val_img_path
            seg_label = params.val_seg_label
            depth_label = params.val_depth_label
            with open(params.val_ann) as json_file:
                self.ann = json.load(json_file)["annotations"]
        elif mode == "test":
            img_path = params.test_img_path
            seg_label = params.test_seg_label
            depth_label = params.test_depth_label
        else:
            ("You entered invalid text !")

        img_names = get_names(img_path, "image")
        seg_names = get_names(seg_label, "label")
        depth_names = get_names(depth_label, "image")
        self.sorted_img = number_names(img_names)
        self.sorted_seg = number_names(seg_names)
        self.sorted_depth = number_names(depth_names)

    def __getitem__(self, index):
        image = cv2.imread(self.sorted_img[index])
        # print(self.sorted_img[index], index)
        image = self.transform(image)
        # print("image : ", torch.max(image), torch.min(image)) # 0~255
        image = self.normalize(image, torch.max(image))
        seg = cv2.imread(self.sorted_seg[index])
        seg = cv2.resize(seg, dsize=(256, 128))
        seg = self.transform(seg)
        seg = seg[0, :, :]
        # print("seg : ", torch.max(seg), torch.min(seg)) # 0~33
        # seg = self.normalize(seg, torch.max(seg))
        depth = cv2.imread(self.sorted_depth[index])
        depth = cv2.resize(depth, dsize=(256, 128))
        depth = self.transform(depth)
        depth = depth[0, :, :].unsqueeze(0)
        depth = self.normalize(depth, torch.max(depth))
        # print("depth : ", torch.max(depth), torch.min(depth)) # 0~126
        bbox = []
        cls = []
        
        if len(self.ann[index]["segments_info"]) == 0:
            bbox = [[-1, -1, -1, -1]]
            cls = [-1]
        else:
            for segment in self.ann[index]["segments_info"]:
                if len(segment) != 0:
                    [x1, y1, w, h] = segment["bbox"]
                    bbox.append([y1, x1, (y1 + h), (x1 + w)])
                    cls.append(segment["category_id"] + 1)
    

        od = {
            # "img_idx": index,
            # "img_size": (256, 128),
            "bbox": torch.FloatTensor(bbox).to(params.device),
            "cls": torch.FloatTensor(cls).to(params.device),
        }
        return {
            "img": image,
            "seg": seg,
            "depth": depth,
            "od": od,
        }  # image = [3,1024,2048], seg = [1024,2048], depth = [1,1024,2048]

    def __len__(self):
        return len(self.sorted_img)

    def transform(self, obj):
        obj = torch.tensor(obj)
        # print(obj)
        obj = obj.type(torch.FloatTensor).to(params.device)
        # obj = obj[:,:,::-1]
        obj = obj.permute(2, 0, 1).contiguous()

        # obj = obj[::-1,:,:]
        # print("Here is obj size: ", obj.size())
        return obj  # [c, h, w]

    # def transforms1(self, obj):
    #     obj = torch.tensor(obj)
    #     obj = obj.type(torch.FloatTensor).to(params.device)

    # def transform(self, obj):
    #     obj = torch.tensor(obj)
    #     # print(obj)
    #     obj = obj.type(torch.FloatTensor).to(params.device)
    #     obj = obj.permute(2,0,1).contiguous()
    #     print("Here is obj size: ", obj.size())
    #     return obj # [c, h, w]

    def normalize(self, obj, val):
        obj = obj / float(val)
        obj = obj.type(torch.cuda.FloatTensor)
        return obj


    def collate_fn(self, samples):
        imgs = [sample["img"] for sample in samples]
        segs = [sample["seg"] for sample in samples]
        depths = [sample["depth"] for sample in samples]
        bboxes = [sample["od"]["bbox"] for sample in samples]
        clss = [sample["od"]["cls"] for sample in samples]
            
        padded_bbox = torch.nn.utils.rnn.pad_sequence(
            bboxes, batch_first=True, padding_value=-1
        )
        padded_cls = torch.nn.utils.rnn.pad_sequence(
            clss, batch_first=True, padding_value=-1
        )
        return {
            "img": torch.stack(imgs).contiguous(),
            "seg": torch.stack(segs).contiguous(),
            "depth": torch.stack(depths).contiguous(),
            "od": {"bbox": padded_bbox.contiguous(), "cls": padded_cls.contiguous()},
        }


if __name__ == "__main__":

    train_dataset = BaseLoader("train")
    val_dataset = BaseLoader("val")
    train_loader = data.DataLoader(
        train_dataset, batch_size=params.batch_size, collate_fn=collate_fn
    )

    save_path = "./show_img"
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    # train_dataset.__getitem__(2970)

    for idx, output in enumerate(train_loader):
        # print(idx)
        path = []
        img = output.get("img")
        seg = output.get("seg")
        depth = output.get("depth")
        path = [img, seg, depth]
        od = output.get("od")
        print(img.size(), od["bbox"].size(), od["cls"])
        break
        visualize(path, save_path)

