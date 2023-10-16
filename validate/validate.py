from omegaconf import DictConfig
from validate_utils import (
    batch_df,
    plot_bboxes,
    plot_cutouts,
    plot_masks,
    save_original_full_res_images,
    validation_sample_df,
)


# # Define directories and load data
def main(cfg: DictConfig) -> None:
    ########### Change these as necessary ###########
    cutout_dir = cfg.data.cutoutdir
    species_info_json = cfg.data.species
    batch_id = cfg.general.batch_id
    batch_dir = cfg.data.batchdir

    #################################################
    # ## Settings
    sample_sz = cfg.validate.sample_sz
    random_state = cfg.validate.random_state
    fig_size = tuple(cfg.validate.figsize)
    dpi = cfg.validate.dpi

    # Full res images
    plot_full_res_images = cfg.validate.plot_full_res_images
    full_res_save_location = cfg.validate.full_res_save_location

    # Bboxes
    plot_boxes = cfg.validate.plot_boxes
    bbox_save_location = cfg.validate.bbox_save_location
    bbox_transparent_fc = cfg.validate.bbox_transparent_fc

    # Masks
    plot_masks_results = cfg.validate.plot_masks
    mask_transparent_fc = cfg.validate.mask_transparent_fc
    mask_save_location = cfg.validate.mask_save_location

    # Cutotus
    plot_cutout_results = cfg.validate.plot_cutouts
    cutout_save_location = cfg.validate.cutout_save_location
    cutout_transparent_fc = cfg.validate.cutout_transparent_fc
    title = True

    #################################################

    # Prep data df
    ogdf = batch_df(batch_id, cutout_dir, batch_dir)
    df = validation_sample_df(ogdf, sample_sz=sample_sz, random_state=random_state)

    if plot_full_res_images:
        save_original_full_res_images(df, save_location=full_res_save_location)

    if plot_boxes:
        plot_bboxes(
            df,
            show_labels=True,
            transparent_fc=bbox_transparent_fc,
            save_location=bbox_save_location,
            axis=False,
            figsize=fig_size,
            dpi=dpi,
        )
    if plot_masks_results:
        plot_masks(
            df,
            figsize=fig_size,
            transparent_fc=mask_transparent_fc,
            save_location=mask_save_location,
            species_info=species_info_json,
            dpi=dpi,
        )
    if plot_cutout_results:
        plot_cutouts(
            df,
            figsize=fig_size,
            save_location=cutout_save_location,
            title=title,
            transparent_fc=cutout_transparent_fc,
            dpi=dpi,
        )
