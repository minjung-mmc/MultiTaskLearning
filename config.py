import torch


class params:
    def __init__(self):
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.train_img_path = "../dataset/CityScape/leftImg8bit/train/"
        self.train_seg_label = "../dataset/CityScape/gtFine/train/"
        self.train_depth_label = "../dataset/CityScape/disparity/train/"
        self.val_img_path = "../dataset/CityScape/leftImg8bit/val/"
        self.val_seg_label = "../dataset/CityScape/gtFine/val/"
        self.val_depth_label = "../dataset/CityScape/disparity/val/"
        self.val_ann = "../dataset/CityScape/customized/cityscapes_panoptic_val.json"
        self.train_ann = (
            "../dataset/CityScape/customized/cityscapes_panoptic_train.json"
        )
        self.batch_size = 1
        self.mode = "Train"
        self.num_epoch = 100
        self.gamma = 0.1
        self.lr = 0.0005
        self.num_seg_step = 10
        self.num_depth_step = 10
        self.num_classes_seg = 34  # 원래 34개
        self.num_classes_depth = 1
        self.beta1 = 0.5


params = params()


if __name__ == "__main__":

    params = params()
    print(params.city)
