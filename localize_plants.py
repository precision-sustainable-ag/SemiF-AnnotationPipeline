from pathlib import Path

import cv2
import pandas as pd
import torch
from omegaconf import DictConfig


def load_model(model_path):
    device = torch.device(0)
    ## load model
    model = torch.hub.load('ultralytics/yolov5',
                           'custom',
                           path=model_path,
                           device=device)
    ## set model inference config and settings
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.15  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    # model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16]
    model.max_det = 1000  # maximum number of detections per image
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    # model.cpu()  # CPU
    return model


def inference(imgpath, model):
    # Load images
    img = cv2.imread(imgpath)[..., ::-1]
    # Get results
    results = model(img, size=640)
    # Convert to pd dataframe
    df = results.pandas().xywhn[0]
    # df = results.pandas().xyxy[0]
    # Add imgfilename to columns
    for i, row in df.iterrows():
        df.at[i, 'imgname'] = imgpath.name
    return df


def main(cfg: DictConfig) -> None:
    ## Define directories
    model_path = cfg.general.model_path
    csv_savepath = cfg.general.detection_csvpath
    imagedir = Path(cfg.general.imagedir)
    ## Get image files
    images = sorted(imagedir.rglob("*.jpg"), reverse=True)
    ## Init model
    model = load_model(model_path)
    # Get images
    dfimgs = []
    for idx, imgp in enumerate(images):
        df = inference(imgp, model)
        dfimgs.append(df)

    # Concat dfs
    appended_data = pd.concat(dfimgs)
    appended_data.to_csv(csv_savepath)
