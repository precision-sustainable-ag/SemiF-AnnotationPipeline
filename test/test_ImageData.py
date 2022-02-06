from datasets import ImageData
from pathlib import Path
import numpy as np


def test_ImageData(test_image_data):
    """Test for LoadImageData class"""
    image = ImageData(test_image_data.upload_id, test_image_data.path,
                      test_image_data.date, test_image_data.time,
                      test_image_data.location, test_image_data.cloud_cover,
                      test_image_data.camera_height,
                      test_image_data.camera_lens, test_image_data.pot_height)

    # Get and parse iterator
    iterimg = next(iter(image))
    # Assert expressions
    assert type(iterimg) == tuple
    img_path, rgbimg, img0, img_cnt_disp = iterimg
    assert Path(img_path).exists()
    assert type(rgbimg) == np.ndarray
    assert type(img0) == np.ndarray
    assert img_cnt_disp is not None
