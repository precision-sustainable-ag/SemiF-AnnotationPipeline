import cv2
import multiprocessing
from utils import read_rgbimg, make_exg, make_otsu, make_kmeans, check_kmeans, reduce_holes, exg_minus_exr
from pathlib import Path


class MaskGenerator(object):
    DISPLAY_NAME = "invalid"
    
    def __init__(self):
        self.meta = None

    def read_image(self, imgpath):
        self.input_img = read_rgbimg(imgpath)
        
    def write_image(self, save_path, img):
        cv2.imwrite(save_path, img)
    
    def process(self):
        pass # override in derived classes to perform an actual segmentation

    def start_pipeline(self, args):        
        self.img_path, self.save_dir, self.viproc, self.clsproc = args
        self.read_image(self.img_path)
        print(f"filename: {Path(self.img_path).name}\npath: {Path(self.img_path).parent}")        
        return self.process()

class ExGMaskGenerator(MaskGenerator):
    DISPLAY_NAME = "ExG"
    def process(self):
        self.vi = make_exg(self.input_img)

        if "OTSU" in str.upper(self.clsproc):
            self.mask = make_otsu(self.vi)

        elif "KMEANS" in str.upper(self.clsproc):
            self.mask = check_kmeans(make_kmeans(self.vi))*255

        self.cleaned_mask = reduce_holes(self.mask) 
        
        save_path = str(Path("data/test_results", Path(self.img_path).name))
        self.write_image(save_path, self.cleaned_mask)
        
        print('ExG mask saved.')

class ExGRMaskGenerator(MaskGenerator):
    DISPLAY_NAME = 'ExGmR'
    def process(self):

        self.vi = exg_minus_exr(self.input_img)
        
        if "OTSU" in str.upper(self.clsproc):
            self.mask = make_otsu(self.vi)
        
        elif "KMEANS" in str.upper(self.clsproc):
            self.mask = check_kmeans(make_kmeans(self.vi))*255

        self.cleaned_mask = reduce_holes(self.mask) 

        save_path = str(Path("data/test_results", f"{Path(self.img_path).stem}.png"))
        self.write_image(save_path, self.cleaned_mask)
        print('Otsu mask saved.')

if __name__ == '__main__':

    from utils import get_imgs
    vi_procedure = 'exg'
    classification_procedure = "otsu"
    
    mask_gen_class = {
        'exg': ExGMaskGenerator,
        'exgr': ExGRMaskGenerator,
    }.get(vi_procedure)
    
    images = get_imgs("data/sample")
    
    images = [str(x) for x in images]
    
    img_dir = Path("test_results") # save_path
    data = [(img, img_dir, vi_procedure, classification_procedure) for img in images]
    pool = multiprocessing.Pool(3)
    

    results = pool.map(mask_gen_class().start_pipeline, data)
    print('##########FINISHED########')