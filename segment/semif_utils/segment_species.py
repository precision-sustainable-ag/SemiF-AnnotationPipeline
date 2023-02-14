import cv2
import numpy as np
from scipy import ndimage as ndi
from semif_utils.utils import (apply_mask, clear_border, make_exg, make_kmeans,
                               otsu_thresh, reduce_holes, thresh_vi)
from skimage import filters
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


class Segment:

    def __init__(self, img, data) -> None:
        self.species = data.species
        self.bbox_size_th = data.bbox_size_th
        self.img = img
        self.bbox = data.bbox
        self.mask = None
        self.props = None
        self.green_per = None
        self.extends_border = None
        self.rgb, self.hex = self.species['rgb'], self.species['hex']
        self.class_id = self.species['class_id']

    def remove_blue(self, mask, thresh=20000):
        """Returns True if "green" False if not"""
        cutout = apply_mask(self.img, mask, "black")
        hsv = cv2.cvtColor(cutout, cv2.COLOR_RGB2HSV)
        lower = np.array([(160 * 180) / 360, (30 * 255) / 100,
                          (20 * 255) / 100])
        upper = np.array([(290 * 180) / 360, (100 * 255) / 100,
                          (90 * 255) / 100])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        diff = np.where(hsv_mask == 1, 0, 1)
        return diff, np.sum(hsv_mask)
        # return True if np.sum(hsv_mask) > thresh else False

    def check_green(self, mask, thresh=20000):
        """Returns True if "green" False if not"""
        cutout = apply_mask(self.img, mask, "black")
        hsv = cv2.cvtColor(cutout, cv2.COLOR_RGB2HSV)
        lower = np.array([40, 70, 120])
        upper = np.array([90, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        return True if np.sum(hsv_mask) > thresh else False, np.sum(hsv_mask)

    def general_seg(self, mode="cluster"):
        exg_vi = make_exg(self.img, thresh=True)
        th_vi = thresh_vi(exg_vi)
        if mode == "cluster":
            temp_mask = make_kmeans(th_vi)
            chk_green, green_sum = self.check_green(temp_mask)
            if not chk_green:  # if false
                # Reverse mask
                temp_mask = np.where(temp_mask == 1, 0, 1)
            temp_mask = reduce_holes(temp_mask * 255) * 255
        elif mode == "threshold":
            temp_mask = otsu_thresh(th_vi)
            temp_mask = reduce_holes(temp_mask * 255) * 255
        self.mask = temp_mask.copy()
        return self.mask

    def multi_otsu(self, img_type="vi", classes=3):
        if img_type == "vi":
            exg_vi = make_exg(self.img, thresh=True)
            img = thresh_vi(exg_vi)

        thresholds = filters.threshold_multiotsu(img, classes=classes)
        mask = np.digitize(img, bins=thresholds)
        return mask

    def watershed(self, img_type="rgb", kernel=(3, 3)):
        if img_type == "rgb":
            img = self.img
        elif img_type == "vi":
            exg_vi = make_exg(self.img, thresh=True)
            img = thresh_vi(exg_vi)

        distance = ndi.distance_transform_edt(img)
        coords = peak_local_max(distance,
                                footprint=np.ones(kernel),
                                labels=img)
        mask_zeros = np.zeros(distance.shape, dtype=bool)
        mask_zeros[tuple(coords.T)] = True
        markers, _ = ndi.label(mask_zeros)
        self.mask = watershed(-distance, markers, mask=img)
        return self.mask

    def contour_seg(self):
        pass

    def cotlydon(self):
        exg_vi = make_exg(self.img, thresh=True)
        th_vi = thresh_vi(exg_vi)
        self.mask = otsu_thresh(th_vi)
        return self.mask

    def lambsquarters(self, img_type="vi", cotlydon=False):
        multi = self.multi_otsu(img_type=img_type, classes=3)
        label_img = label(multi > 0, connectivity=2)
        if cotlydon:
            up_dev = 250
        else:
            props = np.array([
                x.area
                for x in regionprops(label_img, thresh_vi(make_exg(self.img)))
            ])
            up_dev = [
                props.mean() - 3 * props.std(),
                props.mean() + 3 * props.std()
            ][1]
        holed = reduce_holes(label_img,
                             min_object_size=up_dev,
                             min_hole_size=up_dev)
        med = cv2.medianBlur(holed.astype(np.uint8), 1)
        self.mask = np.where(med > 0, 1, 0)
        return self.mask

    def broad_seedling(self):
        pass

    def grass_seedling(self):
        pass

    def broadlead_vegetative(self):
        pass

    def grass_vegetative(self):
        pass

    def get_extends_borders(self):
        self.extends_border = False if np.array_equal(
            self.mask, clear_border(self.mask)) else True
        return self.extends_border

    def is_green(self, percent_thresh=.002):
        # def is_green(green_sum, image_shape, percent_thresh=.2):
        """ Returns true if number of green pixels is 
            above certain threshold percentage based on
            total number of pixels.
        """
        # check threshold value
        assert percent_thresh <= 1, "green sum percent threshold is greater than 1. Must be less than or equal to 1."
        if self.mask.max() == 0:
            return False
        else:
            total_pixels = self.mask.shape[0] * self.mask.shape[1]

            self.green_per = self.props.green_sum / total_pixels

            is_green = True if self.green_per > percent_thresh else False

            return is_green

    def get_bboxarea(self):
        x1, y1, x2, y2 = self.bbox
        width = float(x2) - float(x1)
        length = float(y2) - float(y1)
        area = width * length
        return area

    def rem_bbotblue(self):
        mask = self.mask.copy().astype(np.uint8)
        temp_cutout = cv2.bitwise_and(self.img, self.img, mask=mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        hsv = cv2.cvtColor(temp_cutout, cv2.COLOR_RGB2HSV)
        ## mask of blue (80,55,0) ~ (110, 130, 255)
        hsvmask = cv2.inRange(hsv, (75, 22, 0), (110, 255, 255))
        # Calculate number of hsv pixels
        total_pixs = hsvmask.shape[0] * hsvmask.shape[1]
        blue_prop = hsvmask.sum() / total_pixs
        if blue_prop > .2:
            hsvmask = np.where(hsvmask > 0, 1, 0)
            self.mask = np.where(hsvmask != 0, 0, self.mask).astype(np.uint8)
            return True
        else:
            return False

    def is_grass(self):
        pass

    def is_cotyledon(self):
        pass

    def is_small_bbox(self):
        """
        Return "cotlydon" for growth stage if bounding box is small
        """
        x1, y1, x2, y2 = self.bbox
        width = x2 - x1
        length = y2 - y1
        area = width * length
        return True if area < self.bbox_size_th else False

    def is_rgb_empty(self):
        """ Returns true if rgb crop is empty
        """
        return True if self.img.max() == 0 else False

    def is_mask_empty(self):
        """ Returns true if mask is empty
        """
        return True if self.mask.max() == 0 else False

    def is_below_pot(self):
        """ 
        Returns True if segment is below pot elevation and thus noise
        """
        pass

    def semantic_mask(self):
        pass

    def instance_mask(self):
        pass
