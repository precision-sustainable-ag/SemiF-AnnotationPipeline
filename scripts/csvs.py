import os

# new_dir = os.path.expanduser('~/matt/SemiF-AnnotationPipeline')
# os.chdir(new_dir)

from pathlib import Path
from dataclasses import asdict
import pandas as pd
from multiprocessing import Pool
from semif_utils.utils import get_cutout_meta
from tqdm import tqdm
# def cutoutmeta2csv(cutoutdir, save_df=True):
def cutoutmeta2csv(meta, save_df=True):
    unknonw_cls = {
            "scientific_name": "unknown",
            "common_name": "unknown",
            "USDA_symbol": "unknown",
            "EPPO": "  unknown",
            "authority": "unknown",
            "collection_location": "unknown",
            "polygon_id": ""
        }
    
    ############
    # Get dictionaries
    cutout = asdict(get_cutout_meta(meta))
    row = cutout["cutout_props"]
    cls = cutout["cls"]
    lcontours = cutout["local_contours"]
    # Extend nested dicts to single column header
    for ro in row:
        rec = {ro: row[ro]}
        cutout.update(rec)
    for cl in cls:
        unknowns = ["plant", "dicot"]
        spec = unknonw_cls if cls in unknowns or not (type(cls) is dict) else {cl: cls[cl]}
        cutout.update(spec)
    formatted_conts = []
    for _, list_cont in enumerate(lcontours):
        cont = [item for sublist in list_cont for item in sublist]
        formatted_conts.append(cont)
    cutout["local_contours"] = formatted_conts

    # Remove duplicate nested dicts
    cutout.pop("cutout_props")
    cutout.pop("cls")
    
    # Create and append df
    cutdf = pd.DataFrame(cutout)
    ###########
    return cutdf


if __name__ == "__main__":

    # Get cutout directories based on state_id prefix
    state_ids = ["MD", "NC", "TX"]
    cutout = Path("data/semifield-cutouts/")
    # check if glob results is batch directory, if batch directory is a directory, and if batch directory contains contents. 
    cutoutcsv_stems = [x.stem for x in cutout.glob("*.csv")]
    cutoutdirs = [x for x in cutout.glob("*") if (x.name.split("_")[0] in state_ids) and (x.is_dir()) and (any(Path(x).iterdir())) and (x.name not in cutoutcsv_stems)]
    save_df = True
    multi_proc = True


    for cutoutdir in tqdm(cutoutdirs):
        batch_id = cutoutdir.name
        metas = [x for x in Path(cutoutdir).glob("*.json")]
        
        if multi_proc:
            cpu_count = os.cpu_count() - 1
            with Pool(cpu_count) as pool:
                data = pool.map(cutoutmeta2csv, metas)
            # Concat and reset index of main df
            cutouts_df = pd.concat(data)
            cutouts_df = cutouts_df.reset_index().drop(columns="index")
            pool.join()
            pool.close()

        else:
            data = [cutoutmeta2csv(x) for x in tqdm(metas)]
            dfs = []
            for meta in metas:
                df = cutoutmeta2csv(meta)
                dfs.append(df)
        # Concat and reset index of main df
            cutouts_df = pd.concat(dfs)
            cutouts_df = cutouts_df.reset_index().drop(columns="index")
            # # Save dataframe
        if save_df:
            print(f"Saved at {cutoutdir.parent}/{batch_id}.csv")
            cutouts_df.to_csv(f"{cutoutdir.parent}/{batch_id}.csv", sep='\t', header=True, index=False)
        