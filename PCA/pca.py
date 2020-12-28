import cv2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def ImageReconstruction():

    # Read original images
    image_set = []
    image_size = (100, 100, 3)
    for i in range(34): # For 1 ~ 34
        img = cv2.imread("./PCA/srcfile/%d.jpg" % (i + 1))
        image_set.append(img.flatten())

    # Generate PCA estimator, then fit it
    pca_esti = PCA(n_components=25)
    pca_esti.fit(image_set)

    # Do PCA transform, then reconstruct the image by inverse transform
    img_comp = pca_esti.transform(image_set)
    img_comp_inv = pca_esti.inverse_transform(img_comp)

    # For each component-inversed image, generate reduced images and compute reconstruction error
    reconstruction_error = []
    image_reduced = []
    for i in range(34):
        img_reduced = img_comp_inv[i]
        img_reduced = (img_reduced - img_reduced.min()) * 255 / (img_reduced.max() - img_reduced.min())
        img_reduced = img_reduced.astype(np.uint8)

        # Compute reconstruction error
        error = np.sum((image_set[i] - img_reduced) ** 2)
        reconstruction_error.append(error)

        # Reduced image collcet with reshaping!
        image_reduced.append(img_reduced.reshape(image_size))

    for i, image in enumerate(image_reduced):
        plt.subplot(4, 17, int(i / 17) * 34 + i % 17 + 1)
        # Reshape to original image size, then showout
        plt.imshow(cv2.cvtColor(image_set[i].reshape(image_size), cv2.COLOR_BGR2RGB))
        plt.xticks(())
        plt.yticks(())
        plt.subplot(4, 17, int(i / 17) * 34 + i % 17 + 17 + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.xticks(())
        plt.yticks(())
    plt.show()

    return reconstruction_error

if __name__ == "__main__":
    print(ImageReconstruction())