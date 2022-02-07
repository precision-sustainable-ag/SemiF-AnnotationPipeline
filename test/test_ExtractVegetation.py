from ExtractVegetation import ExtractVegetation
from datasets import ImageData
from utils import show, get_exg_histogram


def test_ExtractVegetation(test_image_data):
    image = ImageData(test_image_data.upload_id, test_image_data.path,
                      test_image_data.date, test_image_data.time,
                      test_image_data.location, test_image_data.cloud_cover,
                      test_image_data.camera_height,
                      test_image_data.camera_lens, test_image_data.pot_height)
    # image = iter(image)
    for path, rgbimg, img0, img_disp in image:
        # path, rgbimg, img0, img_disp    = i
        # print(img_disp)
        # print(rgbimg)
        ext = ExtractVegetation(rgbimg, exg_thresh=0, normalize_exg=False)
        show(ext.otsu)
        # get_exg_histogram(ext.exg)