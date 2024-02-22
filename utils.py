import numpy as np
import matplotlib.pyplot as plt
from skimage import transform


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