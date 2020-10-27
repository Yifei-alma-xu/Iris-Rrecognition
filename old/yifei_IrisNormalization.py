import numpy as np


def iris_normalization(img_gray, pupil, iris):
    M = 64
    N = 512
    rect = np.zeros((M, N))
    x_p = pupil[0][0]
    y_p = pupil[0][1]
    r_p = pupil[0][2]
    x_i = iris[0][0]
    y_i = iris[0][1]
    r_i = iris[0][2]

    X = np.arange(N) / N
    theta = 2 * np.pi * X
    X_pt = x_p + r_p * np.cos(theta)
    Y_pt = y_p + r_p * np.sin(theta)
    X_it = x_p + r_i * np.cos(theta)
    Y_it = y_p + r_i * np.sin(theta)
    Y = np.arange(M) / M

    for i in range(N):
        x_pt = X_pt[i]
        y_pt = Y_pt[i]
        x_it = X_it[i]
        y_it = Y_it[i]

        # Compute the coordinate of corresponding point (x,y) in the original image
        x = np.minimum((x_pt + (x_it - x_pt) * Y).astype(int), 319)
        y = np.minimum((y_pt + (y_it - y_pt) * Y).astype(int), 279)
        rect[:, i] = img_gray[y, x]

    return rect


if __name__ == "__main__":
    from IrisRecognition import load_dataset
    from IrisLocalization import iris_localization
    import matplotlib.pyplot as plt

    x_train_img, _, _, _ = load_dataset()
    img_gray = x_train_img[311]

    pupil, iris = iris_localization(img_gray)

    rect = iris_normalization(img_gray, pupil, iris)
    plt.figure()
    plt.imshow(rect, cmap='gray')
    plt.show()
