from pathlib import Path

import cv2
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm


def load_model(model_path):
    # TODO create option for cpu
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
    model.max_det = 1000  # maximum number of detections per imag
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    # model.cpu()  # CPU
    return model


def inference(imgpath, model, save_detection=False):
    # Load images
    img = cv2.imread(imgpath)  #[..., ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Get results
    results = model(img, size=640)
    # Convert to pd dataframe
    df = results.pandas().xyxy[0]
    # Add imgfilename to columns
    for i, row in df.iterrows():
        df.at[i, 'imgname'] = imgpath.name
    # return df, imgpath, crops
    if save_detection:
        return df, img, imgpath
    else:
        return df, imgpath


def main(cfg: DictConfig) -> None:
    save_detection = cfg.detect.save_detection
    ## Define directories
    model_path = cfg.detect.model_path
    imagedir = Path(cfg.general.imagedir)
    batchdir = Path(cfg.general.batchdir)
    csv_savepath = Path(cfg.general.batchdir, "detections.csv")

    if save_detection:
        # Crop savepath
        cropsave_dir = Path(batchdir, "detection_crops")
        cropsave_dir.mkdir(parents=True, exist_ok=True)

    ## Get image files
    images = sorted(imagedir.rglob("*.jpg"), reverse=True)
    ## Init model
    model = load_model(model_path)
    # Get images
    dfimgs = []
    for idx, imgp in tqdm(enumerate(images),
                          desc="Localizing Plants",
                          colour="#9266c4",
                          total=len(images)):
        if not save_detection:
            df, imgpath = inference(imgp, model, save_detection=save_detection)

        if save_detection:
            df, img, imgpath = inference(imgp,
                                         model,
                                         save_detection=save_detection)
            num = 0
            for row in df.itertuples():
                x1, y1 = int(row.xmin), int(row.ymin)
                x2, y2 = int(row.xmax), int(row.ymax)
                crop = img[y1:y2, x1:x2]
                fname = Path(imgpath).stem + "_" + str(num) + ".jpg"
                cropsave_path = Path(cropsave_dir, fname)
                cv2.imwrite(cropsave_path,
                            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                num += 1
        dfimgs.append(df)

    # Concat dfs
    appended_data = pd.concat(dfimgs)
    appended_data.to_csv(csv_savepath)