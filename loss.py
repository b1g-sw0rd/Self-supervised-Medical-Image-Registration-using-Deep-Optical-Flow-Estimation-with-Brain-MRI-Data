import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor

alphaval = 0.2
device = torch.device('cuda')

def photometric_loss(fixed, warped):
    h, w = warped.shape[2:]
    fixed = F.interpolate(fixed, (h, w), mode='bilinear', align_corners=False)
    p_loss = charbonnier(fixed - warped)
    return torch.sum(p_loss)/fixed.size(0)  # 这里是一张图片的p loss
    # return torch.sum(p_loss) / fixed.numel()  # 这里是每个像素点的p loss

def photometric_loss_3d(fixed, warped):
    p_loss = charbonnier(fixed - warped)
    return torch.sum(p_loss)/fixed.size(0)  # 这里是一张图片的p loss
    # return torch.sum(p_loss) / fixed.numel()  # 这里是每个像素点的p loss


# Ref: https://github.com/ily-R/Unsupervised-Optical-Flow/blob/master/utils.py
def smoothness_loss(flow):
    b, c, h, w = flow.size()
    v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
    h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
    s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
    s_loss = torch.sum(s_loss, dim=1) / 2
    result = torch.sum(s_loss)/b
    return result


def charbonnier(x, alpha=0.25, epsilon=1.e-9):
    out = torch.pow(torch.pow(x, 2) + epsilon**2, alpha)
    return out


def correlation_loss_3d(fixed, warped):
    b, c, h, w, d= warped.size()
    # fixed = F.interpolate(fixed, (h, w, d), mode='bilinear', align_corners=False)
    vx = warped - torch.mean(warped)
    vy = fixed - torch.mean(fixed)
    device = vx.device
    zero = torch.zeros(b, c, h, w, d).to(device)
    if torch.equal(zero, vx) or torch.equal(zero, vy):
        corr = torch.tensor(1.0).to(device)
    else:
        corr = 1/b * torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        #print("非0了，corr：", corr)
    return 1.0-corr

def correlation_loss(fixed, warped):
    b, c, h, w = warped.size()
    fixed = F.interpolate(fixed, (h, w), mode='bilinear', align_corners=False)
    vx = warped - torch.mean(warped)
    vy = fixed - torch.mean(fixed)
    device = vx.device
    zero = torch.zeros(b, c, h, w).to(device)
    if torch.equal(zero, vx) or torch.equal(zero, vy):
        corr = torch.tensor(1.0).to(device)
    else:
        corr = 1/b * torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        #print("非0了，corr：", corr)
    return 1.0-corr

def OFEloss(flow, warped, fixed, lamb_da=0.5, gamma=100.0, zeta=100.0):
    p_loss = 0  # photometric_loss initial
    s_loss = 0  # smoothness_loss initial
    c_loss = 0  # correlation_loss initial
    n = len(flow)
    weights = 0.05 * np.arange(1,n+1)
    weights = torch.from_numpy(weights)
    weights.to(device)
    for i in range(n):
        p_loss += weights[i] * photometric_loss(fixed, warped[i])   # 权重问题 参考https://github.com/ily-R/Unsupervised-Optical-Flow/blob/master/utils.py
        c_loss += weights[i] * correlation_loss(fixed, warped[i])   # 问题： 权重会不会导致过度受一个尺度flow和warped的影响
        # temp = correlation_loss(fixed, warped[i])
        # c_loss += temp
        s_loss += weights[i] * smoothness_loss(flow[i])
    p_loss = 1/n * gamma * p_loss
    c_loss = 1/n * zeta * c_loss
    s_loss = 1/n * lamb_da * s_loss
    loss_all = p_loss +s_loss +c_loss
    return p_loss, c_loss, s_loss, loss_all


def Affloss(warped, fixed, lamb_da=1.0, gamma=1.0):

    p_loss = photometric_loss_3d(fixed, warped)   # 权重问题 参考https://github.com/ily-R/Unsupervised-Optical-Flow/blob/master/utils.py
    c_loss = correlation_loss_3d(fixed, warped)   # 问题： 权重会不会导致过度受一个尺度flow和warped的影响
    p_loss = gamma * p_loss
    c_loss = lamb_da * c_loss
    loss_all = p_loss +c_loss
    return p_loss, c_loss, loss_all


if __name__ == '__main__':
    test1 = torch.rand(10, 1, 256, 256)
    test2 = torch.rand(10, 1, 256, 256)

    print("fc:", photometric_loss(test1, test2), '/n', "size(0):", test1.size(0))

    h, w = test1.shape[2:]
    test2 = F.interpolate(test2, (h, w), mode='bilinear', align_corners=False)
    p_loss = charbonnier(test1 - test2)
    print("numel:", torch.sum(p_loss) / test1.numel())




