from pathlib import Path
import matplotlib.pyplot as plt


def get_files(
        parent_dir,
        extensions=["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]):
    """
    Get files with multiple extensions using Path.glob
    
    Inputs:
    parent_dir   = parent directory of images
    extensions   = list of extensions to include in search

    Returns:
    files        = list of file names

    :param parent_dir: str
    :param extensions: list of str
    :return files: list of str
    """
    # Check file and directory exists
    assert type(extensions) is list, "Input is not a list"
    assert Path(parent_dir).exists(), "Image directory is not valid.adasd"
    files = []
    # Parse locations and collection image files
    for ext in extensions:
        files.extend(Path(parent_dir).rglob(ext))
    return files


def showimg(img):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axes(False)
    plt.show()