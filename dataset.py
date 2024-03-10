from torch.utils import data
import pandas as pd
import os
from torchvision.io import read_image
import glob
import nibabel as nib
import SimpleITK as itk
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import torch
import cv2


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
        img_path_list.extend(glob.glob(img_glob))
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


if __name__ == '__main__':
    z = 1    # # image_glob = os.path.join(image_dir, "*.img")
    # # print(image_glob)
    # # image_name_list = []
    # # image_name_list.extend(glob.glob(image_glob))
    # # print(image_name_list[::])
    # # print(len(image_name_list))
    # img_path_list = []
    # filelist = os.listdir(image_dir)
    # # print(filelist)
    # for i in filelist:
    #     img_glob = os.path.join(image_dir, i, "PROCESSED\MPRAGE\SUBJ_111", "*.img")
    #     img_path_list.extend(glob.glob(img_glob))
    # # print(len(img_path_list))
    # # print(img_path_list[:4])
    # img_pair = []
    # for j in range(0, len(img_path_list)-1, 2):
    #     img_pair.append([img_path_list[j], img_path_list[j+1]])
    #
    # img_pair = img_pair[:3]
    # print('length:', len(img_pair), 'shape:', np.shape(img_pair), 'first item:', img_pair[0])
    # imgA = []
    # imgB = []
    # for pair in img_pair:
    #     n = 160 # 多少层
    #     tempA = nib.load(pair[0])
    #     tempB = nib.load(pair[1])
    #     temp_vol_A = np.asarray(tempA.get_fdata())
    #     temp_vol_A = np.squeeze(temp_vol_A)
    #     temp_vol_B = np.asarray(tempB.get_fdata())
    #     temp_vol_B = np.squeeze(temp_vol_B)
    #     for index in range(n):
    #         imgA.append(temp_vol_A[:, :, index])
    #         imgB.append(temp_vol_B[:, :, index])
    # print(np.shape(imgA), np.shape(imgB))
    # for i in range(len(imgA)):
    #     tempA = imgA[i]
    #     tempB = imgB[i]
    #     cv2.imwrite("D:/test/imgA/" + str(i) + ".jpg", tempA)
    #     cv2.imwrite("D:/test/imgB/" + str(i) + ".jpg", tempB)

    # plt.imshow(imgA[100])
    # plt.show()

    # # img = nib.load(img_path_list[0])
    # # data = np.asarray(img.get_fdata())
    # # data = np.squeeze(data)
    # # print(np.shape(data))
    # # temp = data[:, :, 150]
    # # print(np.shape(temp), type(temp))
    # # temp = transform.resize(temp, (256, 256), anti_aliasing=True)
    # # plt.imshow(temp)
    # # plt.show()
