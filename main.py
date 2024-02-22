from torch.utils import data
import numpy as np
from PIL import Image
import pandas as pd
import os
from torchvision.io import read_image


class brain_dataset(data.Dataset):
    def __int__(self, image_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir  # 图像目录
        self.mri_images = pd.read_csv(image_dir)  # 这里还要改 应该是包含所有图像名的文件
        self.transform = transform

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir + self.mri_images.iloc[index, 0])
        image = read_image(image_path)
        if self.transform is not None:
            image = self.transform(image)  # 对图片进行某些变换

        return image


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


if __name__ == '__main__':
    print_hi('PyCharm')
