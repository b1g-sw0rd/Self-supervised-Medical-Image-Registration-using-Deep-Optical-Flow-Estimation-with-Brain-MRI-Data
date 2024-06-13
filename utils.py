import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import cv2
import sys


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

    def __init__(self, keep_all=False):
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

def makeColorwheel():

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255
    return colorwheel


def computeColor(u, v):
    colorwheel = makeColorwheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img.astype(np.uint8)


def computeImg(flow, verbose=False, savePath=None):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    if flow.shape[0] == 2:
        u = flow[0, :, :]
        v = flow[1, :, :]
    else:
        u = flow[:, :, 0]
        v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    # fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([maxrad, np.amax(rad)])

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = computeColor(u, v)
    if savePath is not None:
        cv2.imwrite(savePath, img)
    if verbose:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def computerArrows(flow, step=16, verbose=False, savePath=None, img=None):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    if img is None:
        vis = np.ones((h, w)).astype('uint8')*255
    else:
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x2, y2), (x1, y1) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    if savePath is not None:
        cv2.imwrite(savePath, vis)
    if verbose:
        cv2.imshow('arrowsViz', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return vis
