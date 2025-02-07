import math
import piexif
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def extract_exif(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Extract EXIF data
    exif_data = image._getexif()
    
    # Convert EXIF tags to readable names
    if exif_data:
        readable_exif = {TAGS.get(tag): value for tag, value in exif_data.items() if tag in TAGS}
        return readable_exif
    else:
        return "No EXIF data found."

def estimate_focal_length_35mm(focal_length, sensor_width, sensor_height):
    # Diagonal size of a 35mm full-frame sensor
    diag_35mm = math.sqrt(36**2 + 24**2)  # Full-frame diagonal in mm (43.27 mm)

    # Diagonal size of the given sensor
    diag_sensor = math.sqrt(sensor_width**2 + sensor_height**2)

    # Estimate Focal Length in 35mm Film format
    focal_length_35mm = focal_length * (diag_35mm / diag_sensor)
    return focal_length_35mm


def update_exif(image_path, focal_length, focal_length_35mm, width, height):
    try:
        # Load the existing EXIF data
        exif_dict = piexif.load(image_path)

        # Update Focal Length and Focal Length in 35mm Film
        exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (int(focal_length * 100), 100)  # Rational number
        exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(focal_length_35mm)  # Integer

        # Update Image Dimensions (Width and Height)
        exif_dict["0th"][piexif.ImageIFD.ImageWidth] = int(width)
        exif_dict["0th"][piexif.ImageIFD.ImageLength] = int(height)

        # Save the updated EXIF data
        exif_bytes = piexif.dump(exif_dict)
        image = Image.open(image_path)
        image.save(image_path, "jpeg", exif=exif_bytes)

        print(f"Updated EXIF data saved to {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images(image_paths, focal_length, focal_length_35mm, width_pix, height_pix):
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for image_path in image_paths:
            futures.append(executor.submit(update_exif, str(image_path), focal_length, focal_length_35mm, width_pix, height_pix))
        
        for future in futures:
            future.result()  # Ensures all tasks complete

if __name__ == "__main__":
    # Example inputs
    focal_length = 60  # Focal length in mm
    sensor_width = 46.2  # Sensor width in mm
    sensor_height = 32.87  # Sensor height in mm
    width_pix = 13376  # Image width in pixels
    height_pix = 9528  # Image height in pixels

    # Estimate the 35mm equivalent focal length
    focal_length_35mm = estimate_focal_length_35mm(focal_length, sensor_width, sensor_height)
    print(f"FocalLengthIn35mmFilm: {focal_length_35mm:.2f} mm")

    image_dir = Path("data/semifield-developed-images/NC_2025-02-03/images")
    images = sorted(image_dir.glob("*.jpg"))

    # process_images(images, focal_length, focal_length_35mm, width_pix, height_pix)

    for image in images:
        # Example usage
        exif_data = extract_exif(image)
        print(f"\nEXIF data for {image.name}:")
        for tag, value in exif_data.items():
            print(f"{tag}: {value}")
        