import numpy as np


def iris_normalization(img, pupil, iris):
    # 64*512 MxN
    M, N = 64, 512
    norm_img = np.zeros((M, N))
    pupil_x, pupil_y, pupil_r  = pupil
    iris_x, iris_y, iris_r = iris
    
    for Y in range(M):
        for X in range(N):
            theta = 2*np.pi*X/N
            yp = pupil_x + pupil_r*np.sin(theta)
            xp = pupil_y + pupil_r*np.cos(theta)

            # get the outer boundary coordinate
            yi = iris_x + iris_r*np.sin(theta)
            xi = iris_y + iris_r*np.cos(theta)

            x = min(int(xp + (xi-xp)*Y/M),319)
            y = min(int(yp + (yi-yp)*Y/M),279)
            
            norm_img[Y][X] = img[y][x]
    
    return norm_img


if __name__ == "__main__":
    from fy_IrisRecognition import load_dataset
    from fy_IrisLocalization import iris_localization
    import matplotlib.pyplot as plt

    x_train_img, _, _, _ = load_dataset()
    img_gray = x_train_img[311]

    pupil, iris = iris_localization(img_gray)

    rect = iris_normalization(img_gray, pupil, iris)
    plt.figure()
    plt.imshow(rect, cmap='gray')
    plt.show()
