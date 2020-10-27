import cv2
import matplotlib.pyplot as plt
import numpy as np


def iris_localization(img_gray):
    # Image blur by kernel
    img_blur = cv2.blur(img_gray, ksize=(15, 15))

    # Detect the edges
    med_val = np.median(img_blur)
    lower = int(max(0, .07 * med_val))
    upper = int(min(255, .1 * med_val))
    edges1 = cv2.Canny(img_blur, threshold1=lower, threshold2=upper)
    edges2 = cv2.Canny(img_blur,
                       threshold1=lower * 2.5,
                       threshold2=upper * 2.5)

    # Find the two circles of pupil and iris by Hough transform
    circle1 = cv2.HoughCircles(edges1,
                               cv2.HOUGH_GRADIENT,
                               1,
                               100,
                               param1=100,
                               param2=20,
                               minRadius=80,
                               maxRadius=120)
    circle2 = cv2.HoughCircles(edges2,
                               cv2.HOUGH_GRADIENT,
                               1,
                               100,
                               param1=100,
                               param2=20,
                               minRadius=20,
                               maxRadius=80)
    iris = circle1[0, :, :]
    iris = np.uint16(np.around(iris))
    pupil = circle2[0, :, :]
    pupil = np.uint16(np.around(pupil))

    return pupil, iris


if __name__ == "__main__":
    from IrisRecognition import load_dataset    
    x_train_img, _, _, _ = load_dataset()

    img_gray = x_train_img[311]

    pupil, iris = iris_localization(img_gray)

    img2 = img_gray.copy()
    cv2.circle(img2, (pupil[0][0], pupil[0][1]), pupil[0][2], (255, 0, 0), 2)
    cv2.circle(img2, (pupil[0][0], pupil[0][1]), 2, (255, 0, 0), 3)
    cv2.circle(img2, (iris[0][0], iris[0][1]), iris[0][2], (255, 0, 0), 2)
    cv2.circle(img2, (iris[0][0], iris[0][1]), 2, (255, 0, 0), 3)
    plt.figure()
    plt.imshow(img2, cmap='gray')
    plt.show()