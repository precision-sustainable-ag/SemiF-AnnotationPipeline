
from skimage.measure import label
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage import filters
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import numpy as np
import cv2

from semif_utils.utils import (make_exg, thresh_vi, make_kmeans, otsu_thresh, reduce_holes, apply_mask)

class Segment:
    def __init__(self, img) -> None:
        self.img = img
    
    def check_color(self, mask, thresh=20000):
        """Returns True if "green" False if not"""
        cutout = apply_mask(self.img, mask, "black")
        hsv = cv2.cvtColor(cutout, cv2.COLOR_RGB2HSV)
        lower = np.array([40, 70, 120])
        upper = np.array([90, 255, 255])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        return True if np.sum(hsv_mask)> thresh else False 

    def general_seg(self, mode="cluster"):
        exg_vi = make_exg(self.img, thresh=True)
        th_vi = thresh_vi(exg_vi)
        if mode == "cluster":
            mask = make_kmeans(th_vi)
            if not self.check_color(mask):
                mask = np.where(mask == 1, 0, 1)
            mask = reduce_holes(mask * 255) * 255
        elif mode == "threshold":
            mask = otsu_thresh(th_vi)
            mask = reduce_holes(mask * 255) * 255
        return mask
    

    def multi_otsu(self, img_type="vi", classes=3):
        if img_type == "vi":
            exg_vi = make_exg(self.img, thresh=True)
            img = thresh_vi(exg_vi)
        
        thresholds = filters.threshold_multiotsu(img, classes=classes)
        regions = np.digitize(img, bins=thresholds)
        return regions

    
    def watershed(self,img_type="rgb", kernel=(3,3)):
        if img_type == "rgb":
            img = self.img
        elif img_type == "vi":
            exg_vi = make_exg(self.img, thresh=True)
            img = thresh_vi(exg_vi)
            
        distance = ndi.distance_transform_edt(img)
        coords = peak_local_max(distance, footprint=np.ones(kernel), labels=img)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=img)
        return labels

    def contour_seg(self):
        pass

    def cotlydon(self):
        exg_vi = make_exg(self.img, thresh=True)
        th_vi = thresh_vi(exg_vi)
        vi_mask = otsu_thresh(th_vi)
        return vi_mask

    def lambsquarters(self, img_type="vi", cotlydon=False):
        multi = self.multi_otsu(img_type=img_type, classes=3)
        label_img = label(multi>0, connectivity=2)
        if cotlydon:
            up_dev = 250
        else:
            props = np.array([x.area for x in regionprops(label_img, thresh_vi(make_exg(self.img)))])
            up_dev = [props.mean() - 3 * props.std(), props.mean() + 3 * props.std()][1]
        holed = reduce_holes(label_img, min_object_size=up_dev, min_hole_size=up_dev)
        med = cv2.medianBlur(holed.astype(np.uint8), 1)
        mask = np.where(med>0, 1, 0)
        return mask
        
    
    def broad_seedling(self):
        pass
    
    def grass_seedling(self):
        pass

    def broadlead_vegetative(self):
        pass

    def grass_vegetative(self):
        pass

