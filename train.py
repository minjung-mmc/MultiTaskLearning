#%%
import torch.optim as optim
import torch.nn as nn
import torch
import random
import cv2
import pickle
from config import params
from doubleUnet.model import doubleUNet

# from utils import visualize, get_path
import utils
import time
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage
import numpy as np
from effdet.efficientdet import EfficientDet
from effdet.config_bifpn import get_efficientdet_config
from effdet import AnchorLabeler, Anchors
from effdet.loss import DetectionLoss
import torch.nn.functional as F
import os
import natsort

config = {
    "name": "tf_efficientdet_d0",
    "backbone_name": "tf_efficientnet_b0",
    "backbone_args": {"drop_path_rate": 0.2},
    "backbone_indices": None,
    "image_size": [1024, 2048],  # about pred # h, w
    "num_classes": 34,
    "min_level": 3,
    "max_level": 7,
    "num_levels": 5,
    "num_scales": 3,
    "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
    "anchor_scale": 4.0,
    "pad_type": "same",
    "act_type": "swish",
    "norm_layer": None,
    "norm_kwargs": {"eps": 0.001, "momentum": 0.01},
    "box_class_repeats": 3,
    "fpn_cell_repeats": 3,
    "fpn_channels": 64,
    "separable_conv": True,
    "apply_resample_bn": True,
    "conv_after_downsample": False,
    "conv_bn_relu_pattern": False,
    "use_native_resize_op": False,
    "downsample_type": "max",
    "upsample_type": "nearest",
    "redundant_bias": True,
    "head_bn_level_first": False,
    "head_act_type": None,
    "fpn_name": None,
    "fpn_config": None,
    "fpn_drop_path_rate": 0.0,
    "alpha": 0.25,
    "gamma": 1.5,
    "label_smoothing": 0.0,
    "legacy_focal": False,
    "jit_loss": False,
    "delta": 0.1,
    "box_loss_weight": 50.0,
    "soft_nms": False,
    "max_detection_points": 5000,
    "max_det_per_image": 100,
    "url": "https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth",
    "load_network": False,
    "load_network_path": "./weights/2021-11-22 19:34",
}
#%%


#%%

writer = SummaryWriter()


class trainer:
    def train(self, data_loader_train, data_loader_val):

        model = self.init_model()
        if config["load_network"] == True:
            net = os.listdir(config["load_network_path"])
            net = natsort.natsorted(net)[-2]
            model.load_state_dict(torch.load(config["load_network_path"] + "/" + net))

        # config = get_efficientdet_config("efficientdet_d0")
        # model = EfficientDet(config, 34, 1, pretrained_backbone=False).to(params.device)

        optimizer = self.init_optimizer(model)

        # optimizer_seg = self.init_optimizer(model)
        # optimizer_depth = self.init_optimizer(model)

        criterion = nn.CrossEntropyLoss()
        criterion1 = nn.MSELoss()

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[params.num_epoch * 0.75],
            gamma=params.gamma,
        )

        ######################################################################
        anchors = Anchors.from_config(config)
        anchor_labeler = AnchorLabeler(
            anchors.to(params.device), config["num_classes"], match_threshold=0.5
        )

        loss_fn = DetectionLoss(config)

        ######################################################################

        save_weight, save_image = utils.get_path()
        for epoch in range(params.num_epoch):
            for idx, output in enumerate(data_loader_train):
                img = output.get("img")
                seg = output.get("seg")
                depth = output.get("depth")
                target = output.get("od")
                # if target["bbox"].shape[1] == 0:
                #     continue

                model.train()  # 혹시 모르니까
                model.zero_grad()
                # print("Device Check: ", img.device)

                #######################################################################
                out_seg, out_depth, class_out, box_out = model(img)
                cls_targets, box_targets, num_positives = anchor_labeler.batch_label_anchors(
                    target["bbox"], target["cls"]
                )
                # print(class_out.device(), box_out.device(), cls_targets.device(), box_targets.device(), num_positives.device())
                loss, class_loss, box_loss = loss_fn(
                    class_out, box_out, cls_targets, box_targets, num_positives
                )
                Loss_seg = criterion(
                    out_seg, seg.long()
                )  # out_seg = [1,34,1024,2048], seg = [1,1024,2048] (0~33)

                Loss_depth = criterion1(out_depth, depth)
                Loss = Loss_seg + 10 * Loss_depth + 0.005 * loss

                ###########################################################################

                ################################ Old #############################################
                # Loss_seg = criterion(
                #     out_seg, seg.long()
                # )  # out_seg = [1,34,1024,2048], seg = [1,1024,2048] (0~33)

                # Loss_depth = criterion1(out_depth, depth)

                # Loss = Loss_seg + 10 * Loss_depth
                ################################ Old #############################################

                # Loss = Loss_seg
                Loss.backward()
                optimizer.step()
                # Loss_depth = criterion

                # for seg_step in range(params.num_seg_step):
                #     model.zero_grad()
                #     out_seg, out_depth = model(img)
                #     pred = torch.argmax(out_seg,1)
                #     Loss_seg = criterion(pred, seg)

                #     Loss_seg.backward()
                #     optimizer_seg.step()

                # for depth_step in range(params.num_depth_step):
                #     model.zero_grad()
                #     out_seg, out_depth = model(img)

                #     Loss_depth = criterion(out_depth, depth).to(params.device)

                #     Loss_depth.backward()
                #     optimizer_depth.step()

                if idx % 100 == 0 or idx == (len(data_loader_train) - 1):
                    bs = img.size(0)
                    # print("model output", box_out[])
                    # out_depth *= 126
                    img *= 255
                    # depth *= 126
                    out_seg = torch.argmax(out_seg, dim=1)  # size = (1024, 2048)
                    out_seg = utils.decode_segmap(out_seg[0])
                    seg = utils.decode_segmap(seg[0])
                    out_seg = ToPILImage()(out_seg.detach())
                    seg = ToPILImage()(seg.detach())  # size = (1024, 2048, 3)
                    out_depth = ToPILImage()(out_depth[0].detach())
                    img = F.interpolate(img, size=(128, 256))
                    img = img[0].permute(1, 2, 0).contiguous().int().to("cpu")
                    img = img.numpy()[:, :, ::-1].copy()
                    depth = ToPILImage()(depth[0].detach())
                    #########################################################################################
                    class_out, box_out, indices, classes = utils.post_process(
                        class_out,
                        box_out,
                        num_levels=config["num_levels"],
                        num_classes=config["num_classes"],
                        max_detection_points=config["max_detection_points"],
                    )

                    img_scale, img_size = (
                        [
                            torch.tensor(1).to(params.device),
                            torch.tensor(1).to(params.device),
                        ],
                        [
                            torch.tensor([256, 128]).to(params.device),
                            torch.tensor([256, 128]).to(params.device),
                        ],
                    )
                    res = utils.batch_detection(
                        bs,
                        class_out,
                        box_out,
                        anchors.boxes,
                        indices,
                        classes,
                        img_scale,
                        img_size,
                        max_det_per_image=config["max_det_per_image"],
                        soft_nms=config["soft_nms"],
                    )
                    bboxes = utils.decode_det(res[0].unsqueeze(0))
                    # print("resshape", res.shape)
                    #########################################################################################

                    imgs = [
                        # out_seg.detach(),
                        img,
                        out_seg,
                        seg,
                        out_depth,
                        depth,
                        img,
                        img,
                    ]

                    utils.visualize(imgs, save_image, bboxes, target, epoch, idx)
                    print(
                        "epoch : {}, index: {},  Loss : {:.4f}, Loss_seg : {:.4f}, Loss_depth : {:.4f}, Loss_cls : {:.4f}, Loss_box : {:.4f}".format(
                            epoch, idx, Loss, Loss_seg, Loss_depth, class_loss, box_loss
                        )
                    )

                    writer.add_scalar(
                        "Loss/train",
                        Loss,
                        epoch * len(data_loader_train) + idx * params.batch_size,
                    )
                    writer.add_scalar(
                        "Loss/train/depth",
                        Loss_depth,
                        epoch * len(data_loader_train) + idx * params.batch_size,
                    )
                    writer.add_scalar(
                        "Loss/train/seg",
                        Loss_seg,
                        epoch * len(data_loader_train) + idx * params.batch_size,
                    )
                    writer.add_scalar(
                        "Loss/train/cls",
                        class_loss,
                        epoch * len(data_loader_train) + idx * params.batch_size,
                    )
                    writer.add_scalar(
                        "Loss/train/box",
                        box_loss,
                        epoch * len(data_loader_train) + idx * params.batch_size,
                    )

            ################################ Validate ######################################
            with torch.set_grad_enabled(False):
                model.eval()
                val_loss = 0
                for i, output in enumerate(data_loader_val):
                    img = output.get("img")
                    seg = output.get("seg")
                    depth = output.get("depth")
                    target = output.get("od")
                    # if target["bbox"].shape[1] == 0:
                    #     continue
                    out_seg, out_depth, class_out, box_out = model(img)

                    cls_targets, box_targets, num_positives = anchor_labeler.batch_label_anchors(
                        target["bbox"], target["cls"]
                    )
                    loss, class_loss, box_loss = loss_fn(
                        class_out, box_out, cls_targets, box_targets, num_positives
                    )
                    Loss_seg = criterion(out_seg, seg.long())
                    Loss_depth = criterion1(out_depth, depth)
                    Loss = Loss_seg + 10 * Loss_depth + 0.005 * loss
                    val_loss += Loss
                val_loss = val_loss / len(data_loader_val)

                writer.add_scalar("Loss/val", val_loss, epoch)

            ################################ Validate ######################################

            # writer.add_scalar("Loss/")
            scheduler.step()
            utils.save_network(model.eval(), save_weight, epoch)
            # scheduler_s.step()
            # scheduler_d.step()

    def init_model(self):
        config = get_efficientdet_config("efficientdet_d0")
        model = EfficientDet(
            config,
            params.num_classes_seg,
            params.num_classes_depth,
            pretrained_backbone=False,
        ).to(params.device)
        # model = FSUNet(params.num_classes_seg).to(params.device)
        # model.apply(self.weights_init)
        model.train()
        return model

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def init_optimizer(self, model):
        optimizer = optim.Adam(
            model.parameters(), params.lr, betas=(params.beta1, 0.999)
        )
        return optimizer

