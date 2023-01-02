import logging

from omegaconf import DictConfig
import traceback

from inspect_utils.inspect_utils import CompileDataframe, CutoutStats, ManualInspector

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    log.info(f"Inspecting cutouts for {cfg.general.season}.")

    cd = CompileDataframe(cfg)

    if cfg.inspect.compile.initiate:
        try:
            log.info(
                "Compiling cutout metadata results into a single dataframe.")
            df = cd.compile_df()
        except Exception as e:
            log.error(f"Could not compile dataframe. Exiting.\n{e}")
            log.error(traceback.format_exc())
            exit(1)
        if cfg.inspect.compile.save:
            try:
                log.info("Saving dataframe.")
                cd.save_cutout_csv()
            except Exception as e:
                log.error(f"Failed to save season csv. Exiting.\n{e}")
                exit(1)

    if cfg.inspect.stats.calculate:
        # # Calculate stats
        cs = CutoutStats(cfg)
        # if cfg.inspect.stats.save:
        # cs.save()

    ## Visually inspect cutouts
    if cfg.inspect.visually_inspect.start:
        mi = ManualInspector(cfg)
        mi.inspect_gui()

    # Graph inspection results

    # if cfg.inspect.graph_inspection: