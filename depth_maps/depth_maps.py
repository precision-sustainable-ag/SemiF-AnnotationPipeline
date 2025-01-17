# fmt: off
import os
from pathlib import Path

import Metashape as ms
import numpy as np


def export_depth(doc, output_folder, upscale_2_fullsize=True, single_band_f32=True, grayscale_8bit=True, grayscale_16bit=True):
        compr_deflate = ms.ImageCompression()
        compr_deflate.tiff_compression = ms.ImageCompression().TiffCompressionDeflate
        # active chunk
        chunk = doc.chunk

        if chunk.transform.scale:
            scale = chunk.transform.scale
        
        count = 0
        for camera in chunk.cameras:
            if camera in chunk.depth_maps.keys():
                # depth = chunk.depth_maps[camera].image()
                depth = chunk.model.renderDepth(camera.transform, camera.sensor.calibration)  # unscaled depth
                
                if upscale_2_fullsize:
                    depth = depth.resize(9504,6336)
                
                if grayscale_8bit or grayscale_16bit:
                    img = np.frombuffer(depth.tostring(), dtype=np.float32)
                    depth_range = img.max() - img.min()
                    img = depth - img.min()
                    img = img * (1. / depth_range)
                    
                    if grayscale_8bit:
                        img_u8 = img.convert("RGB", "U8")
                        img_u8 = 255 - img_u8
                        img_u8 = img_u8 - 255 * (img_u8 * (1 / 255)) # normalized
                        img_u8 = img_u8.convert("RGB", "U8")
                        img_u8.save(output_folder + "/" + camera.label + "_grayscale_8bit.png", compression = compr_deflate)

                    if grayscale_16bit:
                        img_u16 = img.convert("RGB", "U16")
                        img_u16 = 65535 - img_u16
                        img_u16 = img_u16 - 65535 * (img_u16 * (1 / 65535)) # normalized
                        img_u16 = img_u16.convert("RGB", "U16")
                        img_u16.save(output_folder + "/" + camera.label + "_grayscale_16bit.png", compression = compr_deflate)
                
                if single_band_f32:
                    depth = depth.convert(" ","F32")
                    single_band_f32_img = depth * scale
                    single_band_f32_img.save(output_folder + "/" + camera.label + "_single_band_F32.tif", compression = compr_deflate)

                print("Processed depth map(s) for " + camera.label)
                count += 1
        print("Script finished. Total cameras processed: " + str(count))
        print("Depth maps exported to:\n " + output_folder)


project_path = "/home/psa_images/SemiF-AnnotationPipeline/data/semifield-developed-images/MD_2024-06-28/autosfm/project/MD_2024-06-28.psx"
output_folder = Path("test_depthmaps")
output_folder.mkdir(exist_ok=True, parents=True)
doc = ms.Document()
if os.path.exists(project_path):
    doc.open(project_path, read_only=False, ignore_lock=True)

export_depth(
    doc,
    str(output_folder),
    upscale_2_fullsize=True,
    single_band_f32=False, 
    grayscale_8bit=True, 
    grayscale_16bit=False
    )
