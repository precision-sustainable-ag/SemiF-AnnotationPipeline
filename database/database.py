import sqlite3
import json
import glob
import logging
import os
import sys
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
                category TEXT
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
                categories TEXT
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
    def bulk_insert_developed_table(self):

        paths = self.cfg.developed_images.bulk_insert_paths
        json_files = []
        for path in paths:
            json_files.extend([json_file for json_file in glob.glob(path,
                                                                    recursive=True)
                               if os.path.basename(os.path.dirname(
                    os.path.dirname(json_file))) not in self.skip_batches])
        multiproc_input = [(x, self.cfg.developed_images.json_keys) for x
                      in chunk_list(json_files,self.batch_size)]
        log.info(f"Bulk inserting {len(json_files)} images")
        num_processes = cpu_count()
        with Pool(num_processes) as pool:
            res = list(tqdm(pool.map(Database._process_chunk, multiproc_input)))
        data = []
        for item in res:
            data.extend(item)
        log.info(f"data size: {sys.getsizeof(data)}")

def add_cutouts_data(json_file, db_name, table_name):
    """
    Adds data from the JSON file to the existing SQLite table.
    Assumes the table already exists and has matching columns.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    with open(json_file, 'r') as f:
        row = json.loads(f.read())
        # print(row)
        cutout_props_json = json.dumps(row['cutout_props'])
        category_json = json.dumps(row['category'])
        placeholders = ', '.join(['?' for _ in row])
        # print(f"INSERT INTO {table_name} VALUES ({placeholders})",
        #     tuple(row.values()))
        cursor.execute(f"""
                INSERT INTO {table_name} (
                    season, datetime, bbot_version, batch_id, image_id, cutout_id, 
                    cutout_num, cutout_height, cutout_width, lens_model, validated, 
                    cutout_props, category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            row['season'], row['datetime'], row['bbot_version'],
            row['batch_id'],
            row['image_id'], row['cutout_id'], row['cutout_num'],
            row['cutout_height'],
            row['cutout_width'], row['lens_model'], row['validated'],
            cutout_props_json, category_json
        ))
    conn.commit()
    conn.close()


def add_dev_img_data(json_file, db_name, table_name):
    """
    Adds data from the JSON file to the existing SQLite table.
    Assumes the table already exists and has matching columns.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    with open(json_file, 'r') as f:
        row = json.loads(f.read())
        print(row)
        exif_meta = json.dumps(row['exif_meta'])
        camera_info = json.dumps(row['camera_info'])
        annotations = json.dumps(row['annotations'])
        categories = json.dumps(row['categories'])
        placeholders = ', '.join(['?' for _ in row])
        # print(f"INSERT INTO {table_name} VALUES ({placeholders})",
        #     tuple(row.values()))
        cursor.execute(f"""
                INSERT INTO {table_name} (
                     season, datetime, bbot_version,batch_id, image_id,
                     validated, exif_meta, camera_info, annotations, categories
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            row['season'], row['datetime'], row['bbot_version'],
            row['batch_id'],
            row['image_id'], row['validated'],
            exif_meta, camera_info, annotations, categories
        ))
    conn.commit()
    conn.close()


#
# create_tables('/Users/jbshah/_p/test/longterm_images/test.db')
#
# # add_cutouts_data('/Users/jbshah/_p/test/longterm_images/semifield'
# #                   '-cutouts/NC_2023-07-10/NC_1688995625_0.json',
# #                   '/Users/jbshah/_p/test/longterm_images/test.db',
# #                   'semif_cutouts')
#
# add_dev_img_data('/Users/jbshah/_p/test/longterm_images/semifield-developed'
#                  '-images/NC_2023-07-10/metadata/NC_1688995625.json',
#                   '/Users/jbshah/_p/test/longterm_images/test.db',
#                   'semif_developed_images')

def main(cfg: DictConfig) -> None:
    db = Database(cfg.database)
    db.create_developed_table()
    db.create_cutouts_table()

    if cfg.database.bulk_insert:
        db.bulk_insert_developed_table()
