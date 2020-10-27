import cv2
import matplotlib.pyplot as plt
import numpy as np


def iris_localization(img):
    #img = img_gray
    ymax, xmax = img.shape
    ## step 1: project to roughly estimate (Xp, Yp)
    Xp = np.sum(img,0).argmin()
    Yp = np.sum(img,1).argmin()
    ## step 2: binarize a 120x120 region centered at (Xp, Yp)
    crop_xmin, crop_xmax = max(Xp-60, 0), min(Xp+60, xmax)
    crop_ymin, crop_ymax = max(Yp-60, 0), min(Yp+60, ymax)
    img_120 = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    _, img_120_bi = cv2.threshold(img_120,np.median(img_120), 255, 
                                  cv2.THRESH_BINARY)
    # update a more accuarte Xp, Yp
    circles = cv2.HoughCircles(img_120_bi,cv2.HOUGH_GRADIENT,1,100,
                                param1=200,param2=10,
                               minRadius=15,maxRadius=100)
    if circles is not None:
        circle = circles[0,:][0]
        circle = np.uint16(np.around(circle))
        Xp = crop_xmin + circle[0]
        Yp = crop_ymin + circle[1]
    ## step 3: calcuate the exact parameters of the two circles
    ## using edge detection(Canny) and Hough transformation
    new_xmin, new_xmax = max(Xp-140, 0), min(Xp+140, xmax)
    new_ymin, new_ymax = max(Yp-160, 0), min(Yp+160, ymax)
    recentered_img = img[new_ymin:new_ymax, new_xmin:new_xmax]
    blured_copy = cv2.medianBlur(recentered_img, 11)
    mean_val = np.mean(blured_copy)
    lower = int(max(0, 0.66*mean_val))
    upper = int(min(255, 1.33*mean_val))
    #eye_edges1 = cv2.Canny(blured_copy,100,200) 
    #eye_edges2 = cv2.Canny(blured_copy,10,50)
    #eye_edges_iris = cv2.Canny(blured_copy,100,200) 
    eye_edges_pupil = cv2.Canny(blured_copy,lower,upper) 
    eye_edges_iris = cv2.Canny(blured_copy,5,30)
    circles_iris = cv2.HoughCircles(eye_edges_iris,cv2.HOUGH_GRADIENT,1,300,
                            param1=200,param2=20,
                           minRadius=60,maxRadius=130)
    circles_pupil = cv2.HoughCircles(eye_edges_pupil,cv2.HOUGH_GRADIENT,1,300,
                            param1=200,param2=20,
                           minRadius=20,maxRadius=80)
    if circles_pupil is not None:
        circle = circles_pupil[0,:][0]
        pupil_circle = np.uint16(np.around(circle))
        pupil_circle[0] = pupil_circle[0]+new_xmin
        pupil_circle[1] = pupil_circle[1]+new_ymin
    if circles_iris is not None:
        circle = circles_iris[0,:][0]
        iris_circle = np.uint16(np.around(circle))
        iris_circle[0] = iris_circle[0]+new_xmin
        iris_circle[1] = iris_circle[1]+new_ymin
    
    return pupil_circle, iris_circle


if __name__ == "__main__":
    from fy_IrisRecognition import load_dataset    
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