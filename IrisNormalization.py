import numpy as np


def iris_normalization(img, pupil, iris):
    # 64*512 MxN
    M, N = 64, 512
    norm_img = np.zeros((M, N))
    pupil_x, pupil_y, pupil_r  = pupil
    iris_x, iris_y, iris_r = iris
    
    theta = 2*np.pi*np.arange(N)/N
    #X_pt = x_p + r_p * np.cos(theta)
    #Y_pt = y_p + r_p * np.sin(theta)
    #X_it = x_p + r_i * np.cos(theta)
    #Y_it = y_p + r_i * np.sin(theta)
    Y = np.arange(M) / M
    
    yp = pupil_y + pupil_r*np.sin(theta)
    xp = pupil_x + pupil_r*np.cos(theta)

    #yi = iris_y + iris_r*np.sin(theta)
    #xi = iris_x + iris_r*np.cos(theta)
    yi = pupil_y + iris_r*np.sin(theta)
    xi = pupil_x + iris_r*np.cos(theta)
    #for Y in range(M):
    #    x = min(int(xp + (xi-xp)*Y/M),319)
    #    y = min(int(yp + (yi-yp)*Y/M),279)

    #    norm_img[Y][X] = img[y][x]
    
    for i in range(N):
        x_pt = xp[i]
        y_pt = yp[i]
        x_it = xi[i]
        y_it = yi[i]

        # Compute the coordinate of corresponding point (x,y) in the original image
        x = np.minimum((x_pt + (x_it - x_pt) * Y).astype(int), 319)
        y = np.minimum((y_pt + (y_it - y_pt) * Y).astype(int), 279)
        norm_img[:, i] = img[y, x]
        
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
