import cv2
import mahotas as mh
import numpy as np
from skimage.feature import greycomatrix, local_binary_pattern
from skimage.measure import regionprops


def calculate_color_histogram_mean(image_path, bins=256):
    hist_red, hist_green, hist_blue = calculate_color_histogram(image_path,
                                                                bins=256)

    # Calculate the mean histogram values for each channel
    mean_hist_red = np.mean(hist_red)
    mean_hist_green = np.mean(hist_green)
    mean_hist_blue = np.mean(hist_blue)

    return mean_hist_red, mean_hist_green, mean_hist_blue


def calculate_lbp_mean(image_path, radius=1, n_points=8):
    hist_lbp = calculate_lbp(image_path, radius=1, n_points=8)
    # Calculate the mean LBP histogram
    mean_hist_lbp = np.mean(hist_lbp)

    return mean_hist_lbp


def calculate_area_and_aspect_ratio_mean(image_path):
    areas, aspect_ratios = calculate_area_and_aspect_ratio(image_path)

    # Calculate the mean area and mean aspect ratio
    mean_area = np.mean(areas) if len(areas) > 0 else 0.0
    mean_aspect_ratio = np.mean(
        aspect_ratios) if len(aspect_ratios) > 0 else 0.0

    return mean_area, mean_aspect_ratio


def calculate_elongation_and_roundness_mean(image_path):
    elongations, roundnesses = calculate_elongation_and_roundness(image_path)

    # Calculate the mean elongation and mean roundness
    mean_elongation = np.mean(elongations) if len(elongations) > 0 else 0.0
    mean_roundness = np.mean(roundnesses) if len(roundnesses) > 0 else 0.0

    return mean_elongation, mean_roundness


def calculate_eccentricity_and_solidity_mean(image_path):
    eccentricities, solidities = calculate_eccentricity_and_solidity(
        image_path)

    # Calculate the mean eccentricity and mean solidity
    mean_eccentricity = np.mean(
        eccentricities) if len(eccentricities) > 0 else 0.0
    mean_solidity = np.mean(solidities) if len(solidities) > 0 else 0.0

    return mean_eccentricity, mean_solidity


def calculate_edge_density_mean(image_path):
    # Calculate the Canny edge map
    edges, image_gray_shape = calculate_edge_density(image_path)

    # Calculate edge density as the ratio of edge pixels to total pixels
    edge_density = np.sum(edges) / (image_gray_shape[0] * image_gray_shape[1])

    return edge_density


def calculate_haralick_features_mean(image_path):

    # Calculate Haralick texture features
    haralick_features = calculate_haralick_features(image_path)

    # Calculate the mean of each Haralick feature
    mean_haralick_features = np.mean(haralick_features, axis=0)

    return mean_haralick_features


##############################################################################
##############################################################################


def calculate_color_histogram(image_path, bins=256):
    # Load the image
    image = cv2.imread(image_path)

    # Split the image into its individual color channels (BGR order)
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Compute histograms for each color channel
    blue_hist = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])
    green_hist = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
    red_hist = cv2.calcHist([red_channel], [0], None, [256], [0, 256])

    return blue_hist, green_hist, red_hist


def calculate_lbp(image_path, radius=1, n_points=8):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate Local Binary Pattern (LBP) for the image
    lbp_image = local_binary_pattern(image_gray,
                                     n_points,
                                     radius,
                                     method='uniform')

    # Compute the histogram of LBP values
    hist_lbp, _ = np.histogram(lbp_image.ravel(),
                               bins=np.arange(0, n_points + 3),
                               range=(0, n_points + 2))

    # Calculate the mean LBP histogram
    # mean_hist_lbp = np.mean(hist_lbp)

    # return mean_hist_lbp
    return hist_lbp


def calculate_area_and_aspect_ratio(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(image_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store areas and aspect ratios
    areas = []
    aspect_ratios = []

    # Calculate the area and aspect ratio for each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            areas.append(area)
            _, (w, h), _ = cv2.minAreaRect(contour)
            aspect_ratio = float(w) / h
            aspect_ratios.append(aspect_ratio)

    # Calculate the mean area and mean aspect ratio
    cleaned_area = areas if len(areas) > 0 else 0.0
    cleaned_aspect_ratio = aspect_ratios if len(aspect_ratios) > 0 else 0.0

    return cleaned_area, cleaned_aspect_ratio


def calculate_elongation_and_roundness(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(image_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store elongations and roundnesses
    elongations = []
    roundnesses = []

    # Calculate elongation and roundness for each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            _, (w, h), _ = cv2.minAreaRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            elongation = np.sqrt(1 - (aspect_ratio**(-2)))
            elongations.append(elongation)

            perimeter = cv2.arcLength(contour, True)
            roundness = (4 * area) / (np.pi * (perimeter**2))
            roundnesses.append(roundness)

    # Calculate the mean elongation and mean roundness
    cleaned_elongation = elongations if len(elongations) > 0 else 0.0
    cleaned_roundness = roundnesses if len(roundnesses) > 0 else 0.0

    return cleaned_elongation, cleaned_roundness


def calculate_eccentricity_and_solidity(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(image_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store eccentricities and solidities
    eccentricities = []
    solidities = []

    # Calculate eccentricity and solidity for each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

            eccentricity = np.sqrt(1 - circularity)
            eccentricities.append(eccentricity)

            convex_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / convex_area
            solidities.append(solidity)

    # Calculate the mean eccentricity and mean solidity
    cleaned_eccentricity = eccentricities if len(eccentricities) > 0 else 0.0
    cleaned_solidity = solidities if len(solidities) > 0 else 0.0

    return cleaned_eccentricity, cleaned_solidity


def calculate_edge_density(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the Canny edge map
    edges, image_gray_shape = cv2.Canny(image_gray, 100, 200), image_gray.shape

    return edges, image_gray_shape


def calculate_object_count(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary mask
    _, binary_mask = cv2.threshold(image_gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the number of objects (plants) in the image
    object_count = len(contours)

    return object_count


def calculate_haralick_features(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate Haralick texture features
    haralick_features = mh.features.haralick(image_gray)

    return haralick_features


def calculate_entropy(image_path):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the histogram of pixel intensities
    hist, _ = np.histogram(image_gray.ravel(), bins=256, range=[0, 256])

    # Normalize the histogram
    hist = hist.astype("float") / hist.sum()

    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    return entropy


def calculate_glcm(image_path, distances=[1], angles=[0], levels=256):
    # Load the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the GLCM
    glcm = greycomatrix(image_gray,
                        distances=distances,
                        angles=angles,
                        levels=levels)

    return glcm


def calculate_symmetry_score(image_path):
    """A low symmetry score indicates a higher level of symmetry, 
    while a higher score suggests that the left and right halves differ significantly, 
    indicating less symmetry in the image. 
    Keep in mind that this is a basic approach, and depending on your specific use case, 
    you may want to apply additional image preprocessing or use more advanced techniques to calculate a symmetry score.

    Args:
        image_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Calculate the midpoint along the width (x-axis)
    mid_x = width // 2

    # Split the image into left and right halves
    left_half = image[:, :mid_x, :]
    right_half = image[:, mid_x:width, :]

    # Ensure both halves have the same width (in case of odd image width)
    left_half = cv2.resize(left_half,
                           (right_half.shape[1], right_half.shape[0]))

    # Calculate absolute difference between the left and right halves
    diff_image = cv2.absdiff(left_half, right_half)

    # Convert the difference image to grayscale
    diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    # Calculate the symmetry score as the mean intensity of the difference image
    symmetry_score = cv2.mean(diff_gray)[0]

    return symmetry_score


def calculate_grvi(image_path):
    # Load the RGB image
    img = cv2.imread(image_path).astype(np.float32)
    b, g, r = cv2.split(img)

    # Calculate the GRVI index
    grvi = (g - r) / (g + r)

    # Calculate the average GRVI value for the entire image
    average_grvi = np.mean(grvi)

    return average_grvi