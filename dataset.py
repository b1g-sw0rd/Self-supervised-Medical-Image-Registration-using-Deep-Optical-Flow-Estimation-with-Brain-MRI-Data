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
                              Resized, SqueezeDimd, CopyItemsd, ConcatItemsd, RandAffined, DeleteItemsd, Rotate90d)
from monai.data import DataLoader
from monai.inferers import SliceInferer
import random
from utils import seed_everything


def worker_init_fn(worker_id, rank=1, seed=1):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def volume2slices_ds(data_dir, seg_dir,batch_size, val_frac=0.1, test_frac=0.1):
    n_workers = 0
    img_list = sorted(glob(os.path.join(data_dir, '*.img')))
    seg_list = sorted(glob(os.path.join(seg_dir, '*.img')))
    img_dict = [{'image': image, 'seg': seg} for image, seg in  zip(img_list, seg_list)]
    img_dict = img_dict[:10]  # 用于测试前n个vol
    seed_everything(6)
    # 随机划分数据集
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
    train_patch_ds_len = len(train_vol)*80
    val_patch_ds_len = len(val_vol)*80
    transform_vol = Compose(
        [
            LoadImaged(keys=['image','seg'], reader="NibabelReader", image_only=True),  # 做分类，这里image需要加载
            Transposed(keys=['image','seg'], indices=[3, 2, 0, 1]),
            SpatialCropd(keys=['image','seg'], roi_start=(60, 0, 0), roi_end=(140, 176, 208))
            #SpatialCropd(keys="image", roi_start=(119,0,0), roi_end=(179,256,128)), # 截取中间脑部组织 179-119=60层 RAW image
            #ThresholdIntensityd(keys='image',threshold=400, above=True, cval=0.0) # RAW image
        ]
    )

    train_vol_ds = monai.data.CacheDataset(data=train_vol, transform=transform_vol, num_workers=n_workers)
    val_vol_ds = monai.data.CacheDataset(data=val_vol, transform=transform_vol, num_workers=n_workers)
    test_vol_ds = monai.data.CacheDataset(data=test_vol, transform=transform_vol, num_workers=n_workers)

    patch_func = monai.data.PatchIterd(
        keys=['image','seg'],
        patch_size=(1, None,None),  # dynamic last two dimensions
        start_pos=(0, 0, 0)
    )

    patch_transform = Compose(
        [
            SqueezeDimd(keys=['image','seg'], dim=0),  # squeeze the last dim
            Resized(keys=['image','seg'], spatial_size=[256, 256], mode=["bilinear","nearest"]),
            Rotate90d(keys=['image','seg'], spatial_axes=(0,1)),
            CopyItemsd(keys=['image','seg'], times=1, names=["image_t","seg_t"]),
            Rand2DElasticd(keys=["image_t","seg_t"], prob=1, spacing=(16,16), magnitude_range=(0,0.5),
                            padding_mode="zeros", mode=[3,"nearest"]),  # 参数需要设置
            ConcatItemsd(keys=["image", "image_t"], name="image_c", dim=0),
            ConcatItemsd(keys=["seg", "seg_t"], name="seg_c", dim=0),
            DeleteItemsd(keys=["image", "image_t", "seg", "seg_t"]),
            ScaleIntensityd(keys='image_c', minv=0.0, maxv=1.0),
        ]
    )
    train_patch_ds = monai.data.GridPatchDataset(data=train_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)
    val_patch_ds = monai.data.GridPatchDataset(data=val_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)
    test_patch_ds = monai.data.GridPatchDataset(data=test_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)

    train_loader = DataLoader(train_patch_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_patch_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_patch_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader, train_patch_ds_len, val_patch_ds_len


def volume_ds(data_dir, batch_size, val_frac=0.1, test_frac=0.1):
    n_workers = 0
    img_list = sorted(glob(os.path.join(data_dir, '*.img')))
    # seg_list = sorted(glob(os.path.join(seg_dir, '*.img')))
    # img_dict = [{'image': image, 'seg': seg} for image, seg in zip(img_list, seg_list)]
    img_dict = [{'image': image} for image in img_list]
    img_dict = img_dict[:20]  # 用于测试前n个vol

    # 随机划分数据集
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
    train_ds_len = len(train_vol)
    val_ds_len = len(val_vol)
    # transform_vol = Compose(
    #     [
    #         LoadImaged(keys=['image','seg'], reader="NibabelReader", image_only=True),  # 做分类，这里image需要加载
    #         Transposed(keys=['image','seg'], indices=[3, 1, 0, 2]),
    #         SpatialCropd(keys=['image','seg'], roi_start=(0, 0, 60), roi_end=(208, 176, 140)),
    #         # SqueezeDimd(keys=['image'], dim=0),
    #         Resized(keys=['image','seg'], spatial_size=(256, 256,80), mode=["bilinear","nearest"]),
    #         Rotate90d(keys=['image','seg'], k=2, spatial_axes=(0,1)),
    #         CopyItemsd(keys=['image','seg'], times=1, names=["image_t","seg_t"]),
    #         RandAffined(keys=["image_t","seg_t"], prob=1, rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
    #                     translate_range=(10,10,10),scale_range=(0.1, 0.1, 0.1), padding_mode="zeros",mode=["bilinear","nearest"]),
    #         ConcatItemsd(keys=["image", "image_t"], name="image_c", dim=0),
    #         ConcatItemsd(keys=["seg", "seg_t"], name="seg_c", dim=0),
    #         DeleteItemsd(keys=["image", "image_t","seg", "seg_t"]),
    #         ScaleIntensityd(keys='image_c', minv=0.0, maxv=1.0),
    #     ]
    # )

    transform_vol = Compose(
        [
            LoadImaged(keys=['image'], reader="NibabelReader", image_only=True),  # 做分类，这里image需要加载
            Transposed(keys=['image'], indices=[3, 1, 0, 2]),
            # SpatialCropd(keys=['image'], roi_start=(0, 0, 60), roi_end=(208, 176, 140)),
            # SqueezeDimd(keys=['image'], dim=0),
            Resized(keys=['image'], spatial_size=(256, 256, 176), mode=["bilinear"]),
            Rotate90d(keys=['image'], k=2, spatial_axes=(0, 1)),
            CopyItemsd(keys=['image'], times=1, names=["image_t"]),
            RandAffined(keys=["image_t"], prob=1, rotate_range=(0, 0, np.pi / 4),
                        translate_range=(5, 5, 0), padding_mode="zeros",
                        mode=["bilinear"]),
            ConcatItemsd(keys=["image", "image_t"], name="image_c", dim=0),
            DeleteItemsd(keys=["image", "image_t"]),
            ScaleIntensityd(keys='image_c', minv=0.0, maxv=1.0),
        ]
    )

    train_vol_ds = monai.data.CacheDataset(data=train_vol, transform=transform_vol, num_workers=n_workers)
    val_vol_ds = monai.data.CacheDataset(data=val_vol, transform=transform_vol, num_workers=n_workers)
    test_vol_ds = monai.data.CacheDataset(data=test_vol, transform=transform_vol, num_workers=n_workers)

    train_loader = DataLoader(train_vol_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_vol_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_vol_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader, train_ds_len, val_ds_len

def eval_random_ds(data_dir, seg_dir,batch_size):
    n_workers = 0
    img_list = sorted(glob(os.path.join(data_dir, '*.img')))
    seg_list = sorted(glob(os.path.join(seg_dir, '*.img')))
    img_dict = [{'image': image, 'seg': seg} for image, seg in  zip(img_list, seg_list)]
    img_dict = img_dict[:5]  # 用于测试前n个vol

    # 随机划分数据集


    eval_patch_ds_len = len(img_dict)*80
    transform_vol = Compose(
        [
            LoadImaged(keys=['image','seg'], reader="NibabelReader", image_only=True),  # 做分类，这里image需要加载
            Transposed(keys=['image','seg'], indices=[3, 2, 0, 1]),
            SpatialCropd(keys=['image','seg'], roi_start=(60, 0, 0), roi_end=(140, 176, 208))
            #SpatialCropd(keys="image", roi_start=(119,0,0), roi_end=(179,256,128)), # 截取中间脑部组织 179-119=60层 RAW image
            #ThresholdIntensityd(keys='image',threshold=400, above=True, cval=0.0) # RAW image
        ]
    )

    eval_vol_ds = monai.data.CacheDataset(data=img_dict, transform=transform_vol, num_workers=n_workers)


    patch_func = monai.data.PatchIterd(
        keys=['image','seg'],
        patch_size=(1, None,None),  # dynamic last two dimensions
        start_pos=(0, 0, 0)
    )
    seed_everything(8)
    patch_transform = Compose(
        [
            SqueezeDimd(keys=['image','seg'], dim=0),  # squeeze the last dim
            Resized(keys=['image','seg'], spatial_size=[256, 256], mode=["bilinear","nearest"]),
            Rotate90d(keys=['image','seg'], spatial_axes=(0,1)),
            CopyItemsd(keys=['image','seg'], times=1, names=["image_t","seg_t"]),
            Rand2DElasticd(keys=["image_t","seg_t"], prob=1, spacing=(16,16), magnitude_range=(0.5,1),
                            padding_mode="zeros", mode=[3,"nearest"]),  # 参数需要设置
            ConcatItemsd(keys=["image", "image_t"], name="image_c", dim=0),
            ConcatItemsd(keys=["seg", "seg_t"], name="seg_c", dim=0),
            DeleteItemsd(keys=["image", "image_t", "seg", "seg_t"]),
            ScaleIntensityd(keys='image_c', minv=0.0, maxv=1.0),
        ]
    )
    eval_patch_ds = monai.data.GridPatchDataset(data=eval_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)

    eval_loader = DataLoader(eval_patch_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)


    return eval_loader, eval_patch_ds_len


def real_ds(fiximg_dir, fixseg_dir, movimg_dir, movseg_dir, batch_size):
    n_workers = 0
    fiximg_list = sorted(glob(os.path.join(fiximg_dir, '*.img')))
    fixseg_list = sorted(glob(os.path.join(fixseg_dir, '*.img')))
    movimg_list = sorted(glob(os.path.join(movimg_dir, '*.img')))
    movseg_list = sorted(glob(os.path.join(movseg_dir, '*.img')))
    fiximg_list = len(movimg_list)*fiximg_list
    fixseg_list = len(movimg_list) * fixseg_list

    img_dict = [{'fiximg': fiximg, 'fixseg': fixseg, 'movimg': movimg, 'movseg': movseg}
                for fiximg, fixseg, movimg, movseg in  zip(fiximg_list, fixseg_list, movimg_list, movseg_list)]
    # img_dict = img_dict[:1]  # 用于测试前n个vol

    # 随机划分数据集


    eval_patch_ds_len = len(img_dict)*80
    transform_vol = Compose(
        [
            LoadImaged(keys=['fiximg', 'fixseg', 'movimg', 'movseg'], reader="NibabelReader", image_only=True),  # 做分类，这里image需要加载
            Transposed(keys=['fiximg', 'fixseg', 'movimg', 'movseg'], indices=[3, 2, 0, 1]),
            SpatialCropd(keys=['fiximg', 'fixseg', 'movimg', 'movseg'], roi_start=(60, 0, 0), roi_end=(140, 176, 208))
            #SpatialCropd(keys="image", roi_start=(119,0,0), roi_end=(179,256,128)), # 截取中间脑部组织 179-119=60层 RAW image
            #ThresholdIntensityd(keys='image',threshold=400, above=True, cval=0.0) # RAW image
        ]
    )

    eval_vol_ds = monai.data.CacheDataset(data=img_dict, transform=transform_vol, num_workers=n_workers)


    patch_func = monai.data.PatchIterd(
        keys=['fiximg', 'fixseg', 'movimg', 'movseg'],
        patch_size=(1, None,None),  # dynamic last two dimensions
        start_pos=(0, 0, 0)
    )
    seed_everything(6)
    patch_transform = Compose(
        [
            SqueezeDimd(keys=['fiximg', 'fixseg', 'movimg', 'movseg'], dim=0),  # squeeze the last dim
            Resized(keys=['fiximg', 'fixseg', 'movimg', 'movseg'], spatial_size=[256, 256], mode=["bilinear","nearest","bilinear","nearest"]),
            Rotate90d(keys=['fiximg', 'fixseg', 'movimg', 'movseg'], spatial_axes=(0,1)),
            ConcatItemsd(keys=['fiximg', 'movimg'], name="image_c", dim=0),
            ConcatItemsd(keys=['fixseg', 'movseg'], name="seg_c", dim=0),
            DeleteItemsd(keys=['fiximg', 'fixseg', 'movimg', 'movseg']),
            ScaleIntensityd(keys='image_c', minv=0.0, maxv=1.0),
        ]
    )
    eval_patch_ds = monai.data.GridPatchDataset(data=eval_vol_ds, patch_iter=patch_func, transform=patch_transform,
                                           with_coordinates=False)

    eval_loader = DataLoader(eval_patch_ds, batch_size=batch_size, num_workers=n_workers, pin_memory=True, worker_init_fn=worker_init_fn)


    return eval_loader, eval_patch_ds_len


if __name__ == '__main__':
    fiximg_dir = "C:/OASIS1/fiximg"
    fixseg_dir = "C:/OASIS1/fixseg"
    movimg_dir = "C:/OASIS1/movimg"
    movseg_dir = "C:/OASIS1/movseg"

    seed_everything(6)

    eval_generator, eval_length = real_ds(fiximg_dir=fiximg_dir, fixseg_dir=fixseg_dir,
                                          movimg_dir=movimg_dir, movseg_dir=movseg_dir,batch_size=1)
    check_data = monai.utils.misc.first(eval_generator)
    print("first batch's shape img: ", check_data["image_c"].shape)
    print("first batch's shape img: ", check_data["seg_c"].shape)
    img = check_data["image_c"]
    seg = check_data["seg_c"]
    plt.figure("real",(12,8))
    plt.subplot(1,2,1)
    plt.title("fixed")
    plt.imshow(seg[0, 0, :, :], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("moving")
    plt.imshow(seg[0, 1, :, :], cmap="gray")
    plt.show()


    # data_dir = "C:/OASIS1/masked"
    # seg_dir = "C:/OASIS1/seg"
    #
    # seed_everything(36)
    # # train_generator, val_generator, test_generator, train_length, val_length= volume2slices_ds(data_dir=data_dir, seg_dir=seg_dir, batch_size=16)
    # train_generator, val_generator, test_generator, train_length, val_length = volume_ds(data_dir=data_dir,batch_size=2)
    #
    # # i = 0
    # # for imgs in train_generator:
    # #     print(i,':',imgs["image_c"].shape)
    # #     i+=1
    #
    # check_data = monai.utils.misc.first(test_generator)
    # print("first batch's shape img: ", check_data["image_c"].shape)
    # # print("first batch's shape seg: ", check_data["seg_c"].shape)
    # # ma = torch.max(check_data["image_c"])
    # # print(train_length)
    # img = check_data["image_c"]
    # plt.figure("test_3d",(12,8))
    # plt.subplot(1,2,1)
    # plt.title("fixed")
    # plt.imshow(img[0, 0, :, :,0], cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("moving")
    # plt.imshow(img[0, 1, :, :, 0], cmap="gray")
    # plt.show()
    # 查看切片
    # for i in range(5):
    #     s = i
    #     img = check_data["image_c"]
    #     seg = check_data["seg_c"]
    #     plt.figure("visualize", (12, 8))
    #     plt.subplot(3,2,1)
    #     plt.title("img1")
    #     plt.imshow(img[s, 0, :, :], cmap="gray")
    #     plt.subplot(3, 2, 2)
    #     plt.title("seg1")
    #     plt.imshow(seg[s, 0, :, :], cmap="gray")
    #     plt.subplot(3, 2, 3)
    #     plt.title("img2")
    #     plt.imshow(img[s, 1, :, :], cmap="gray")
    #     plt.subplot(3, 2, 4)
    #     plt.title("seg1")
    #     plt.imshow(seg[s, 1, :, :], cmap="gray")
    #     plt.subplot(3, 2, 5)
    #     plt.title("img_diff")
    #     plt.imshow(img[s, 1, :, :] - img[s, 0, :, :], cmap="gray")
    #     plt.subplot(3, 2, 6)
    #     plt.title("seg_diff")
    #     plt.imshow(seg[s, 1, :, :] - seg[s, 0, :, :], cmap="gray")
    #     plt.show()

    # print(torch.max(image1[s, 0, :, :]),torch.min(image1[s, 0, :, :]))


    # # 移动raw文件夹下img/hdr到目标文件夹
    # rootdir = "D:/OASIS"
    # rootlist = os.listdir(rootdir)
    # desdir = "D:/OASIS1/seg"
    # for i in range(len(rootlist)):
    #     path = os.path.join(rootdir, rootlist[i], 'FSL_SEG')
    #     path_glob = os.path.join(path, '*.hdr')
    #     filetemp = glob(path_glob)
    #     for j in range(len(filetemp)):
    #         path_full = os.path.join(path, filetemp[j])
    #         shutil.move(path_full, desdir)
