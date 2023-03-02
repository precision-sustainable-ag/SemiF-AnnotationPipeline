import logging

import hydra
from omegaconf import DictConfig

# sys.path.append("move_data")
from move_data.utils.list_batches import ListBatches

log = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    lb = ListBatches(cfg)

    if cfg.movedata.find_missing.az_list:
        # List blob container contents
        try:
            log.info("Creating a list of the blob container items.")
            lb.az_list()
        except Exception as e:
            log.exception(f"Failed to create container item list. Exiting.")
            exit(1)

    if cfg.movedata.find_missing.organize_temp:
        # Clean azcopy list results
        try:
            log.info("Preping azcopy text results.")
            lb.organize_temp()
        except Exception as e:
            log.exception(f"Failed to prep results. Exiting.")
            exit(1)

    if cfg.movedata.find_missing.find_missing:
        # Get missing dataframe
        print(cfg.movedata.find_missing.container_list)
        try:
            log.info("Finding missing items.")
            df = lb.find_missing()
        except Exception as e:
            log.exception(f"Failed to sort missing items. Exiting.")
            exit(1)

    if cfg.movedata.find_missing.write_missing:
        # Write back to missing batch file location
        try:
            log.info(f"Writing missing items to {cfg.logs.unprocessed}.")
            with open(cfg.logs.unprocessed, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{row.batch}: {', '.join(sorted(row.missing))}\n")
        except Exception as e:
            log.exception(f"Failed to write missing items. Exiting.")
            exit(1)

if __name__ == "__main__":
    main()