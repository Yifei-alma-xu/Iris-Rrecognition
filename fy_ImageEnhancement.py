import cv2
import numpy as np


# Use the histogram equalization to enhance the image
def image_enhancement(img, norm_img):
    # background illumination: 16x16
    img_bg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            block = img[i:i+16, j:j+16]
            #block = block.astype('uint8')
            img[i, j] = np.mean(block)
    
    resized_bg = cv2.resize(img_bg, (norm_img.shape[1], norm_img.shape[0]), 
                         interpolation=cv2.INTER_CUBIC)

    enhanced_light_img = norm_img - resized_bg
    # histogram equalization in each 32x32 region
    #kernel = morp.disk(32)
    #enhanced_img = rank.equalize(enhanced_light_img.astype(np.uint8), 
    #                          selem=kernel)
    #enhanced_img = cv2.GaussianBlur(img_local, (5, 5), 0)
    enhanced_img = enhanced_light_img.copy()
    for i in range(enhanced_img.shape[0]):
        for j in range(enhanced_img.shape[1]):
            block = enhanced_img[i:i+32, j:j+32]
            block = block.astype('uint8')
            dest = cv2.equalizeHist(block)
            enhanced_img[i:i+32, j:j+32] = dest            
    
    return enhanced_img


if __name__ == "__main__":
    from fy_IrisRecognition import load_dataset
    from fy_IrisLocalization import iris_localization
    from fy_IrisNormalization import iris_normalization
    import matplotlib.pyplot as plt

    x_train_img, _, _, _ = load_dataset()
    img_gray = x_train_img[311]

    pupil, iris = iris_localization(img_gray)

    rect = iris_normalization(img_gray, pupil, iris)
    rect_enhanced = image_enhancement(img_gray, rect)

    plt.figure()
    plt.imshow(rect_enhanced, cmap='gray')
    plt.show()
