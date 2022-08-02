
from semif_utils.utils import (make_exg, make_exg_minus_exr, make_exr,
                               make_kmeans, make_ndi, otsu_thresh, parse_dict,
                               reduce_holes, rescale_bbox)

from semif_utils.segment_utils import (ClassifyMask, GenCutoutProps,
                                       VegetationIndex, get_image_meta,
                                       get_watershed, prep_bbox,
                                       seperate_components, thresh_vi)

class Segment:
    def __init__(self, img) -> None:
        self.img = img

    def cotlydon(self):
        exg_vi = make_exg(self.img, thresh=True)
        th_vi = thresh_vi(exg_vi)
        vi_mask = otsu_thresh(th_vi)
        return vi_mask

    def lambsquarters(self):
        """Use Lab"""
        pass
        
    
    def broad_seedling(self):
        pass
    
    def grass_seedling(self):
        pass

    def broadlead_vegetative(self):
        pass

    def grass_vegetative(self):
        pass

