from torch.utils import data
import pandas as pd
import os
from torchvision.io import read_image
from glob import glob
import nibabel as nib
import SimpleITK as itk
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import torch
import cv2
import monai
import shutil
from monai.transforms import (Compose, LoadImaged, Rand2DElasticd, ScaleIntensityd, SpatialCropd, Transposed,
                              Resized, SqueezeDimd, CopyItemsd, ConcatItemsd, Rotate90d, DeleteItemsd)
from monai.data import DataLoader
from monai.inferers import SliceInferer


def img_pair(img_dir):
    # image_glob = os.path.join(image_dir, "*.img")
    # print(image_glob)
    # image_name_list = []
    # image_name_list.extend(glob.glob(image_glob))
    # print(image_name_list[::])
    # print(len(image_name_list))
    img_path_list = []
    filelist = os.listdir(img_dir)
    # print(filelist)
    for i in filelist:
        img_glob = os.path.join(img_dir, i, "PROCESSED\MPRAGE\SUBJ_111", "*.img")
        img_path_list.extend(glob(img_glob))
    # print(len(img_path_list))
    # print(img_path_list[:4])
    img_pair = []
    for j in range(0, len(img_path_list)-1, 2):
        img_pair.append([img_path_list[j], img_path_list[j+1]])

    img_pair = img_pair[:3]
    imgA = []
    imgB = []
    for pair in img_pair:
        n = 160 # 多少层
        tempA = nib.load(pair[0])
        tempB = nib.load(pair[1])
        temp_vol_A = np.asarray(tempA.get_fdata())
        temp_vol_A = np.squeeze(temp_vol_A)
        temp_vol_B = np.asarray(tempB.get_fdata())
        temp_vol_B = np.squeeze(temp_vol_B)
        for index in range(n):
            imgA.append(temp_vol_A[:, :, index])
            imgB.append(temp_vol_B[:, :, index])
    return torch.from_numpy(np.array(imgA)).float(), torch.from_numpy(np.array(imgB)).float()
    # temp = data[:, 128, :] # y代表垂直于脖子 先测试pipeline 用的256x256 后续还需要改回去resize


class brain_dataset(data.Dataset):
    def __init__(self, imgA_dir, imgB_dir):
        self.imgA_dir = imgA_dir  # 图像目录
        self.imgB_dir = imgB_dir
        self.imgA_name = os.listdir(imgA_dir)
        self.imgB_name = os.listdir(imgB_dir)

    def __len__(self):
        return len(self.imgA_name)

    def __getitem__(self, index):
        # if self.transform is not None:
        #     image = self.transform(image)  # 对图片进行某些变换
        imgA = cv2.imread(self.imgA_dir + "/" + str(self.imgA_name[index]), 0)
        imgB = cv2.imread(self.imgA_dir + "/" + str(self.imgA_name[index]), 0)
        imgA = torch.from_numpy(imgA)
        imgB = torch.from_numpy(imgB)
        imgA = torch.unsqueeze(imgA, dim=0)
        imgB = torch.unsqueeze(imgB, dim=0)
        img = torch.cat((imgA, imgB), dim=0)
        return img


def volume2slices_ds(data_dir, batch_size, val_frac=0.1, test_frac=0.1):
    img_list = sorted(glob(os.path.join(data_dir, '*.img')))
    img_dict = [{'image': image} for image in img_list]
    img_dict = img_dict[:10]  # 用于测试前n个vol

    # 随机划分数据集
    np.random.seed(1)  # 用于复现
    length = len(img_dict)
    indices = np.arange(length)
    np.random.shuffle(indices)

    test_split = int(test_frac * length)
    val_split = int(val_frac * length) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]

    train_vol = [img_dict[i] for i in train_indices]
    val_vol = [img_dict[i] for i in val_indices]
    test_vol = [img_dict[i] for i in test_indices]
    train_patch_ds_len = len(train_vol)*60
    val_patch_ds_len = len(val_vol)*60
    transform_vol = Compose(
        [
            LoadImaged(keys='image', reader="NibabelReader", image_only=True),  # 做分类，这里image需要加载
            Transposed('image', [3, 1, 0, 2]),
            SpatialCropd(keys="image", roi_start=(119,0,0), roi_end=(179,256,128)), # 截取中间脑部组织 179-119=60层
            ScaleIntensityd('image')
        ]
    )

    train_vol_ds = monai.data.CacheDataset(data=train_vol, transform=transform_vol, num_workers=6)
    val_vol_ds = monai.data.CacheDataset(data=val_vol, transform=transform_vol, num_workers=6)
    test_vol_ds = monai.data.CacheDataset(data=test_vol, transform=transform_vol, num_workers=6)

    patch_func = monai.data.PatchIterd(
        keys=["image"],
        patch_size=(1, None, None),  # dynamic last two dimensions
        start_pos=(0, 0, 0)
    )

    patch_transform = Compose(
        [
            SqueezeDimd(keys=["image"], dim=0),  # squeeze the last dim
            Resized(keys=["image"], spatial_size=[256, 256], mode="bilinear"),
            CopyItemsd(keys="image", times=1, names=["image_t"]),
            Rand2DElasticd(keys="image_t", prob=1, spacing=(64,64), magnitude_range=(0,1),
                           padding_mode="zeros", mode="bilinear"),  # 参数需要设置
            ConcatItemsd(keys=["image", "image_t"], name="image_c", dim=0),
            DeleteItemsd(keys=["image", "image_t"])
        ]
    )
    train_patch_ds = monai.data.GridPatchDataset(data=train_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)
    val_patch_ds = monai.data.GridPatchDataset(data=val_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)
    test_patch_ds = monai.data.GridPatchDataset(data=test_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)

    train_loader = DataLoader(train_patch_ds, batch_size=batch_size, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_patch_ds, batch_size=batch_size, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_patch_ds, batch_size=batch_size, num_workers=6, pin_memory=True)

    return train_loader, val_loader, test_loader, train_patch_ds_len, val_patch_ds_len



if __name__ == '__main__':
    data_dir = "C:/OASIS1/RAW"
    train_generator, val_generator, test_generator, train_length, val_length= volume2slices_ds(data_dir=data_dir, batch_size=24)
    # i = 0
    # for imgs in train_generator:
    #     print(i,':',imgs["image_c"].shape)
    #     i+=1

    check_data = monai.utils.misc.first(train_generator)
    print("first batch's shape: ", check_data["image_c"].shape)
    print(train_length)

    # 查看切片
    s = 10
    image1 = check_data["image_c"]
    plt.figure("visualize", (8, 8))
    plt.subplot(1,2,1)
    plt.title("image1")
    plt.imshow(image1[s, 0, :, :], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("image2")
    plt.imshow(image1[s, 1, :, :], cmap="gray")
    plt.show()

    # 移动raw文件夹下img/hdr到目标文件夹
    # rootdir = "D:/OASIS"
    # rootlist = os.listdir(rootdir)
    # desdir = "D:/OASIS1/RAW"
    # for i in range(len(rootlist)):
    #     path = os.path.join(rootdir, rootlist[i], 'RAW')
    #     path_glob = os.path.join(path, '*.hdr')
    #     filetemp = glob(path_glob)
    #     for j in range(len(filetemp)):
    #         path_full = os.path.join(path, filetemp[j])
    #         shutil.move(path_full, desdir)
    #         print(path_full)
    #         print(desdir)