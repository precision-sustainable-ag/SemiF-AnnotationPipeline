from utils import (check_kmeans, make_exg, make_kmeans, make_otsu, read_img,
                   reduce_holes)


class GenerateMask:

    def __init__(self, imgpath, species):
        self.imgpath = imgpath
        self.species = species
        self.otsu_list = [
            "clover", "grasses", "goosefoot", "grass", "clover", "grasses",
            "goosefoot", "grass", "clover", "grasses", "goosefoot", "grass"
        ]  # Temporary test list
        self.img = self.read_img()
        self.vi = self.produce_vi()
        self.mask = self.classify_vi()
        self.cleaned_mask = self.clean_mask()
        self.img_meta = None

    def read_img(self):
        return read_img(self.imgpath)

    def produce_vi(self, normalize=False, thresh=0):
        return make_exg(self.img)

    def classify_vi(self):
        if self.species in self.otsu_list:
            return make_otsu(self.vi)
        else:
            return check_kmeans(make_kmeans(self.vi)) * 255

    def clean_mask(self):
        return reduce_holes(self.mask)
