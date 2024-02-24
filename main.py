from torch.utils import data
import numpy as np
from PIL import Image
import pandas as pd
import os
from torchvision.io import read_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv, predict_flow, deconv, crop_like
from torch.nn.init import kaiming_normal_, constant_


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


# 测试
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入8×8×1图像，1个零填充，经过4个3×3×1的卷积核，输出8×8×4的图像
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        # 输入8×8×4图像，经过8个5×5×4卷积核，输出4×4×8的图像
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5)
        # 输入4×4×8图像，输出1×32全连接层
        self.fc1 = nn.Linear(4 * 4 * 8, 32)
        # 接着输出1个参数
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 4 * 4 * 8)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# flownetSimple
class flowNetS(nn.Module):
    def __init__(self, batchNorm=True):
        super(flowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2


if __name__ == '__main__':
    # 构建数据集 Dataloader
    train_dataset = 1
    val_dataset = 2
    train_dataloader = 1
    val_dataloader = 2
    batch_size = 32
    model = flowNetS()
    # param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},{'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]
    lr = 1e-4
    loss_fn = nn.MSELoss()  # 需要修改
    optimizer = torch.optim.Adam(flowNetS.parameters(), lr, betas=(0.9, 0.999))
    net = flowNetS.cuda()
    epochs = 10
    for epoch in range(epochs):
        for x, y in train_dataloader:
            x = x.cuda()
            y = y.cuda()
            outputs = net(x)
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.cuda()
                y = y.cuda()
                net.eval()
                outputs = net(x)
        print(epoch)

