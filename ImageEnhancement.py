import cv2


# Use the histogram equalization to enhance the image
def image_enhancement(rect, img_gray):
    rect = rect.astype(img_gray.dtype)
    rect_enhanced = cv2.equalizeHist(rect)
    return rect_enhanced


if __name__ == "__main__":
    from IrisRecognition import load_dataset
    from IrisLocalization import iris_localization
    from IrisNormalization import iris_normalization
    import matplotlib.pyplot as plt

    x_train_img, _, _, _ = load_dataset()
    img_gray = x_train_img[311]

    pupil, iris = iris_localization(img_gray)

    rect = iris_normalization(img_gray, pupil, iris)
    rect_enhanced = image_enhancement(rect, img_gray)

    plt.figure()
    plt.imshow(rect_enhanced, cmap='gray')
    plt.show()
