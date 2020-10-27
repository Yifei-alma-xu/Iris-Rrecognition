import numpy as np
from scipy import ndimage


def defined_filter(sigma_x, sigma_y, size, f):
    filt = np.zeros((size, size))
    # size should be an odd number
    k = int((size - 1) / 2)
    for x in range(-k, k + 1, 1):
        for y in range(-k, k + 1, 1):
            M1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))
            G = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
                -0.5 * (x**2 / sigma_x**2 + y**2 / sigma_y**2)) * M1
            filt[x, y] = G
    return filt


def feature_extraction(rect_roi):
    m = 280
    filt1 = defined_filter(3, 1.5, 7, 0.07)
    filt2 = defined_filter(4.5, 1.5, 7, 0.07)
    img_filt1 = ndimage.convolve(rect_roi, filt1, mode='wrap')
    img_filt2 = ndimage.convolve(rect_roi, filt2, mode='wrap')
    v = []
    for i in range(64):
        for j in range(6):
            m1 = abs(img_filt1[j * 8:j * 8 + 8, i * 8:i * 8 + 8]).mean()
            s1 = abs(abs(img_filt1[j * 8:j * 8 + 8, i * 8:i * 8 + 8]) -
                     m).mean()
            m2 = abs(img_filt2[j * 8:j * 8 + 8, i * 8:i * 8 + 8]).mean()
            s2 = abs(abs(img_filt2[j * 8:j * 8 + 8, i * 8:i * 8 + 8]) -
                     m).mean()
            v.append(m1)
            v.append(s1)
            v.append(m2)
            v.append(s2)
    return v


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    filt = defined_filter(3, 1.5, 7, 0.07)

    plt.figure()
    plt.imshow(filt, cmap='gray')
    plt.show()