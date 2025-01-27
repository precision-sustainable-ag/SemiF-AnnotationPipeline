import sqlite3
import json
import glob
import logging
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from omegaconf import DictConfig
from tqdm import tqdm

from utils.utils import chunk_list

log = logging.getLogger(__name__)


class Database:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.db_name = cfg.db_path
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()

        self.skip_batches = cfg.skip_batches
        self.batch_size = cfg.batch_size

        self.cutouts_table = cfg.cutouts.table_name
        self.dev_img_table = cfg.developed_images.table_name

    def __del__(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def _check_table(self, table_name):
        check_table_query = f"PRAGMA table_info({table_name});"
        self.cursor.execute(check_table_query)
        table_info = self.cursor.fetchall()
        if table_info:
            get_row_count = f"select count(*) from {table_name}"
            self.cursor.execute(get_row_count)
            row_count = self.cursor.fetchone()[0]
            log.info(f"Table {table_name} has {row_count} rows")
            return True
        else:
            return False

    # added _check_table instead of create table if not exists to identify
    # separate states and also get the count before/after insertions
    def create_cutouts_table(self):
        if not self._check_table(self.cutouts_table):
            cutouts_table = f"""
            CREATE TABLE {self.cutouts_table} (
                season TEXT,
                datetime TEXT,
                bbot_version TEXT,
                batch_id TEXT,
                image_id TEXT,
                cutout_id TEXT primary key ,
                cutout_num INTEGER,
                cutout_height INTEGER,
                cutout_width INTEGER,
                lens_model TEXT,
                validated BOOLEAN,
                cutout_props TEXT,
                category TEXT,
                cutout_version TEXT
            );
            """
            log.info(f"Table {self.cutouts_table} created")
            self.cursor.execute(cutouts_table)
            self.connection.commit()

    def create_developed_table(self):
        if not self._check_table(self.dev_img_table):
            developed_table = f"""
            CREATE TABLE {self.dev_img_table} (
                season TEXT,
                datetime TEXT,
                bbot_version TEXT,
                batch_id TEXT,
                image_id TEXT PRIMARY KEY,
                validated BOOLEAN,
                exif_meta TEXT,
                camera_info TEXT,
                annotations TEXT,
                categories TEXT,
                version TEXT
            );
            """
            self.cursor.execute(developed_table)
            log.info(f"Table {self.dev_img_table} created")
            self.connection.commit()

    @staticmethod
    def _process_chunk(multiprocessing_input):
        data = []
        chunk = multiprocessing_input[0]
        json_keys = multiprocessing_input[1]
        for json_file in chunk:
            with open(json_file) as f:
                row = json.load(f)
                for json_key in json_keys:
                    row[json_key] = json.dumps(row[json_key])
                data.append(row)
        return data

    def bulk_insert(self, table_name, paths, json_keys):
        json_files = []
        for path in paths:
            log.info(f"listing json files in {path}")
            json_files.extend([json_file for json_file in glob.glob(path,
                                                                    recursive=True)
                               if os.path.basename(os.path.dirname(
                    os.path.dirname(json_file))) not in self.skip_batches])
        multiproc_input = [(x, json_keys) for x
                           in chunk_list(json_files, self.batch_size)]
        num_processes = cpu_count()
        log.info(f"Reading {len(json_files)} records using {num_processes} "
                 f"processes")
        with Pool(num_processes) as pool:
            res = list(tqdm(pool.map(Database._process_chunk, multiproc_input)))

        if table_name == self.dev_img_table:
            for item in tqdm(res, desc=f"{self.batch_size} images: "):
                self._insert_dev_images(table_name, item)
        elif table_name == self.cutouts_table:
            for item in tqdm(res, desc=f"{self.batch_size} cutouts: "):
                self._insert_cutouts(table_name, item)

    def _insert_one_dev_image(self, table_name, row):
        try:
            self.cursor.execute(f"""
                            INSERT INTO {table_name} (
                                season, datetime, bbot_version, batch_id, image_id, cutout_id, 
                                cutout_num, cutout_height, cutout_width, lens_model, validated, 
                                cutout_props, category, cutout_version
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                row['season'], row['datetime'], row['bbot_version'],
                row['batch_id'], row['image_id'], row['validated'],
                row['exif_meta'], row['camera_info'],
                row['annotations'],
                row['categories'], row['version']
            ))
        except sqlite3.Error as e:
            log.error(
                f"{row['image_id']}, {row['batch_id']}, {row['cutout_id']} - {e}")

    def _insert_dev_images(self, table_name, data):
        if self.batch_size < 10000:
            for row in tqdm(data):
                self._insert_one_dev_image(table_name, row)
            self.connection.commit()
        else:
            bulk_data = [list(i.values()) for i in data]
            try:
                self.connection.execute("BEGIN TRANSACTION")
                self.cursor.executemany(f"""
                                INSERT INTO {table_name} (
                                    season, datetime, bbot_version,batch_id, image_id,
                                    validated, exif_meta, camera_info, 
                                    annotations, categories, version
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, bulk_data)
            except sqlite3.Error as e:
                self.connection.rollback()
                log.warning(
                    f"Error bulk inserting, inserting {len(bulk_data)} rows individually")
                for row in tqdm(data):
                    self._insert_one_dev_image(table_name, row)
                self.connection.commit()
        self.connection.commit()

    def _insert_one_cutout(self, table_name, row):
        try:
            self.cursor.execute(f"""
                            INSERT INTO {table_name} (
                                season, datetime, bbot_version, batch_id, image_id, cutout_id, 
                                cutout_num, cutout_height, cutout_width, lens_model, validated, 
                                cutout_props, category, cutout_version
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                row['season'], row['datetime'], row['bbot_version'],
                row['batch_id'],
                row['image_id'], row['cutout_id'], row['cutout_num'],
                row['cutout_height'], row['cutout_width'], row['lens_model'],
                row['validated'], row['cutout_props'], row['category'],
                row['cutout_version']
            ))
        except sqlite3.Error as e:
            log.error(
                f"{row['image_id']}, {row['batch_id']}, {row['cutout_id']} - {e}")

    def _insert_cutouts(self, table_name, data):
        if self.batch_size < 10000:
            for row in tqdm(data):
                self._insert_one_cutout(table_name, row)
            self.connection.commit()
        else:
            bulk_data = [list(i.values()) for i in data]
            try:
                self.connection.execute("BEGIN TRANSACTION")
                self.cursor.executemany(f"""
                                        INSERT INTO {table_name} (
                                            season, datetime, bbot_version, batch_id, image_id, cutout_id, 
                                            cutout_num, cutout_height, cutout_width, lens_model, validated, 
                                            cutout_props, category, cutout_version
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                                """, bulk_data)
                self.connection.commit()
            except sqlite3.Error as e:
                self.connection.rollback()
                log.warning(
                    f"Error bulk inserting, inserting {len(bulk_data)} rows individually")
                for row in tqdm(data):
                    self._insert_one_cutout(table_name, row)
                self.connection.commit()


def main(cfg: DictConfig) -> None:
    db = Database(cfg.database)
    db.create_developed_table()
    db.create_cutouts_table()

    if cfg.database.bulk_insert:
        # db.bulk_insert_developed_table()

        developed_images_cfg = cfg.database.developed_images
        db.bulk_insert(developed_images_cfg.table_name,
                       developed_images_cfg.bulk_insert_paths,
                       developed_images_cfg.json_keys)

        cutouts_cfg = cfg.database.cutouts
        db.bulk_insert(cutouts_cfg.table_name,
                       cutouts_cfg.bulk_insert_paths,
                       cutouts_cfg.json_keys)
