import sqlite3
import json
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path):
        self.db_name = db_path
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
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
            log.info(
                f"Table {table_name} already exists, with {row_count} rows")
            return True
        else:
            return False

    # added _check_table instead of create table if not exists to identify
    # separate states and also get the count before/after insertions
    def create_cutouts_table(self, table_name):
        if not self._check_table(table_name):
            cutouts_table = f"""
            CREATE TABLE {table_name} (
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
            log.info(f"Table {table_name} created")
            self.cursor.execute(cutouts_table)
            self.connection.commit()

    def create_developed_table(self, table_name):
        if not self._check_table(table_name):
            developed_table = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
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
            self.connection.commit()

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

def main(cfg:DictConfig) -> None:
    db = Database(cfg.database.db_path)

    developed_img_table = cfg.database.table_names.developed_images
    cutouts_table = cfg.database.table_names.cutouts
    db.create_developed_table(developed_img_table)
    db.create_cutouts_table(cutouts_table)

