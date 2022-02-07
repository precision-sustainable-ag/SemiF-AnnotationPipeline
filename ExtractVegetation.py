import numpy as np
import cv2


class ExtractVegetation:
    """ Extracts vegetation segments
    Inputs:
    data            = Color image (np.array) as an ImageData instance
    exg_thresh      = exg threshold value
    Returns:
    index_array     = Cutout images as an instance of ImageData

    :param rgb_img: np.array
    :return index_array: np.array
    """

    def __init__(self, rgbimg, exg_thresh=None, normalize_exg=True):
        self.rgbimg = rgbimg
        self.normalize_exg = normalize_exg
        self.exg_thresh = exg_thresh
        self.exg = self.make_exg()
        self.otsu = self.otsu_thresh()

    def make_exg(self):
        """Excess Green Index.
        r = R / (R + G + B)
        g = G / (R + G + B)
        b = B / (R + G + B)
        EGI = 2g - r - b
        The theoretical range for ExG is (-1, 2).
        Inputs:
        rgb_img      = Color image (np.array)
        Returns:
        index_array    = Index data as a Spectral_data instance
        :param rgb_img: np.array
        :return index_array: np.array
        """
        # exg_thresh = True
        img = self.rgbimg.astype(float)

        blue = img[:, :, 2]
        green = img[:, :, 1]
        red = img[:, :, 0]
        total = red + green + blue
        if self.normalize_exg:
            exg = 2 * (green / total) - (red / total) - (blue / total)
        else:
            exg = 2 * green - red - blue

        if self.exg_thresh is not None:
            exg = np.where(exg < self.exg_thresh, 0, exg).astype(
                'uint8')  # Thresholding removes low negative values
        return exg

    def otsu_thresh(self, kernel_size=(3, 3)):
        mask_blur = cv2.GaussianBlur(self.exg, kernel_size, 0).astype('uint8')
        ret3, mask_th3 = cv2.threshold(mask_blur, 0.9, 2,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask_th3

    def kmeans(self):
        print()
