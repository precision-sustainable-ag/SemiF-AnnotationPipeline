import pytest


class TestImageData:

    def __init__(self):
        """Test data for ImageData"""
        self.upload_id = "fake_test_id"
        self.path = "test/testdata"
        self.date = "12122022"
        self.time = "010212"
        self.location = "test_TX"
        self.cloud_cover = "test partly cloudy"
        self.camera_height = 1
        self.camera_lens = 35
        self.pot_height = 0.2


@pytest.fixture(scope="session")
def test_image_data():
    """Test data object for ImageData."""
    return TestImageData()