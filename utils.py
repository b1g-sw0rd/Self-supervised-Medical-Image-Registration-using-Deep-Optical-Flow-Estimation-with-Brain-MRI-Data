import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import matplotlib.cm as cm
import torch
import random
from skimage.measure import find_contours
from scipy import spatial
from scipy.spatial.distance import cdist
from torchmetrics.functional.clustering import mutual_info_score
from torchmetrics.functional.regression import pearson_corrcoef
import cv2


def grid_generator():
    result = torch.zeros(256, 256)
    x = torch.from_numpy(np.arange(7, 255, 16))
    y = torch.from_numpy(np.arange(7, 255, 16))
    for i in x:
        result[i, :] = 1.0
    for j in y:
        result[:, j] = 1.0
    return result


def flow_mag(flow):
    ux = flow[0, 0, :, :]
    uy = flow[0, 1, :, :]
    mag = torch.sqrt(ux ** 2 + uy ** 2)
    mag_min = mag.min()
    mag_max = mag.max()
    mag_norm = (mag - mag_min) / (mag_max - mag_min) * 255
    mag_norm = mag_norm * (-1)
    mag_norm = mag_norm + 255
    mag_norm = mag_norm.numpy().astype(np.uint8)
    color_image = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
    colored_image_tensor = torch.from_numpy(color_image).permute(2, 0, 1)  # 调整维度顺序以匹配CHW格式
    return torch.sum(mag) ,torch.unsqueeze(colored_image_tensor,dim=0)


def MSE(fixed, warped):
    return torch.mean(torch.pow(warped-fixed,2))


def PSNR(fixed, warped):
    mse = MSE(fixed,warped)
    if mse < 1.0e-10:
        return 100
    return 10 * torch.log10(1.0 ** 2 / mse)


def MI(fixed, warped):
    fixed = torch.round(fixed*1500).int()
    warped = torch.round(warped*1500).int()
    return mutual_info_score(fixed.reshape(-1), warped.reshape(-1))


def CORR(fixed, warped):
    return pearson_corrcoef(fixed.reshape(-1), warped.reshape(-1))


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (torch.sum(y_true_f) + torch.sum(y_pred_f))


def seg_trans(seg, target):
    temp = seg.clone().detach()
    temp[temp != target] = 0
    temp[temp != 0] = 1

    return temp


def dice_average(y_true, y_pred):
    dice = []
    for i in range(3):
        dice.append(dice_coefficient(seg_trans(y_true, i + 1), seg_trans(y_pred, i + 1)).detach().cpu().numpy())
    return np.mean(dice)


def tensor2img(img_tensor):
    img = img_tensor.numpy()
    img = np.transpose(img_tensor, [1, 2, 0])
    plt.imshow(img)


def normalize(x):
    x = np.float32(x)
    xmin = np.amin(x)
    xmax = np.amax(x)
    b = 1.  # max value (17375)
    a = 0.  # min value (0)
    if (xmax - xmin) == 0:
        out = x
    else:
        out = a + (b - a) * (x - xmin) / (xmax - xmin)
    return out


def rescale_img(img, img_size):
    contrast = np.random.uniform(low=0.7, high=1.3)
    brightness = np.random.normal(0, 0.1, 1)
    img = img*contrast + brightness
    r_img = transform.resize(img, img_size, anti_aliasing=True)
    return normalize(r_img).reshape(1, 256, 256, 1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        # return input[:, :, :target.size(2), :target.size(3)]
        return input


# https://github.com/ily-R/Unsupervised-Optical-Flow/blob/master/unsup_train.py
class averageMeter(object):

    def __init__(self, keep_all=True):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        if self.data is not None:
            self.data.append(value)
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def extract_boundary_points(mask):
    """
    提取二值图像mask的边界点。

    参数:
    - mask: 2D numpy array, 二值图像mask

    返回:
    - points: 边界点的列表 [(x1, y1), (x2, y2), ...]
    """
    contours = find_contours(mask, level=0.5)
    if len(contours) == 0:
        return np.empty((0, 2), dtype=int)
    points = np.vstack(contours).astype(int)

    return points


def get_avg_of_min_hausdorff_distance(lA, lB):
    min_distance = []
    for list_from_lA in lA:
        min_val = None
        for list_from_lB in lB:
            dis = spatial.distance.euclidean(list_from_lA, list_from_lB)
            if min_val is None:
                min_val = dis
                break
            if dis < min_val:
                min_val = dis
        min_distance.append(min_val)
    return np.average(min_distance)

def modified_hausdorff(A,B):
    """compute the 'modified' Hausdorff distance between two
    point sets as described in
    M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance
    for object matching. In ICPR94, pages A:566-568, Jerusalem, Israel,
    1994.
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    """
    D = cdist(A,B)
    fhd = np.mean(np.min(D,axis=0))
    rhd = np.mean(np.min(D,axis=1))
    return max(fhd,rhd)

def dist_hausdorff(seg1, seg2):
    dist = []
    for i in range(3):
        points1 = extract_boundary_points(seg_trans(seg1, i + 1).detach().cpu().numpy())
        points2 = extract_boundary_points(seg_trans(seg2, i + 1).detach().cpu().numpy())
        # d1 = directed_hausdorff(points1, points2)[0]
        # d2 = directed_hausdorff(points2, points1)[0]
        # dist.append(max(d1, d2))
        dist.append(modified_hausdorff(points1,points2))
        # dice_coefficient(seg_trans(y_true, i + 1), seg_trans(y_pred, i + 1)).detach().cpu().numpy()

    return np.mean(dist)

if __name__=="__main__":
    torch.manual_seed(88)
    fixed = torch.rand(256,256)
    warped = torch.rand(256, 256)
    flow = torch.rand(1,2,256,256)








    # 可视化
    # plt.imshow(colored_image_tensor.permute(1, 2, 0).numpy())  # 调整维度顺序以匹配HWC格式
    # plt.axis('off')
    # plt.show()
