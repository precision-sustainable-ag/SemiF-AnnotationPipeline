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
from typing import Any, List, Dict, Tuple
from utils.utils import chunk_list

log = logging.getLogger(__name__)


class Database:
    """
    Class to handle database operations including table creation,
    data insertion, and cleanup for developed and cutouts.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the database connection and configuration.

        Args:
            cfg (DictConfig): The configuration object containing database parameters.
        """
        self.cfg = cfg
        self.db_name = cfg.db_path
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()

        self.skip_batches = cfg.skip_batches
        self.batch_size = cfg.batch_size

        self.cutouts_table = cfg.cutouts.table_name
        self.dev_img_table = cfg.developed_images.table_name

    def __del__(self) -> None:
        """
        Destructor to clean up the database connection.
        Commits changes, vacuums the database, and logs the DB size.
        """
        log.info(f"DB size uncleaned: {os.path.getsize(self.db_name)}")
        if self.connection:
            self.connection.commit()
            self.cursor.execute("vacuum;")
            self.connection.commit()
            self.connection.close()
            self.connection = None
            log.info(f"DB size cleaned: {os.path.getsize(self.db_name)}")

    def _check_table(self, table_name: str) -> bool:
        """
        Check if the table exists in the database and log the row count.
        """
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
    def create_cutouts_table(self) -> None:
        """
        Create the cutout table if it does not already exist.
        """
        if not self._check_table(self.cutouts_table):
            # Define the SQL query to create the cutout table
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

    def create_developed_table(self) -> None:
        """
        Create the developed images table if it does not already exist.
        """
        if not self._check_table(self.dev_img_table):
            # SQL query to create the developed table
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
    def _process_chunk(multiprocessing_input: Tuple[List[str], List[str]]) -> List[Dict[str, Any]]:
        """
        Process a chunk of JSON files by loading each file and serializing
        specified keys using JSON dumps.

        Args:
            multiprocessing_input (Tuple[List[str], List[str]]):
                A tuple where the first element is a list of JSON file paths and
                the second element is a list of keys to be JSON-serialized.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the processed JSON data.
        """
        data = []
        chunk = multiprocessing_input[0]
        json_keys = multiprocessing_input[1]
        # Process each JSON file in the chunk
        for json_file in chunk:
            with open(json_file) as f:
                row = json.load(f)
                # Serialize the specified keys to JSON strings
                for json_key in json_keys:
                    row[json_key] = json.dumps(row[json_key])
                data.append(row)
        return data

    def bulk_insert(self, table_name: str, paths: List[str], json_keys: List[str]) -> None:
        """
        Perform a bulk insertion of JSON records into the specified table.
        Uses multiprocessing to parse JSON files in chunks before insertion.

        Args:
            table_name (str): The name of the table to insert data into.
            paths (List[str]): List of file path patterns to search for JSON files.
            json_keys (List[str]): List of keys whose values should be serialized.
        """
        json_files = []
        # Loop over all the paths to collect the JSON files, applying some logic to skip certain batches
        for path in paths:
            if table_name == self.dev_img_table:
                log.info(f"listing json files in {path}")
                json_files.extend([json_file for json_file in glob.glob(path,
                                                                        recursive=True)
                                   if os.path.basename(os.path.dirname(
                        os.path.dirname(json_file))) not in self.skip_batches])
            elif table_name == self.cutouts_table:
                log.info(f"listing json files in {path}")
                json_files.extend([json_file for json_file in glob.glob(path,
                                                                        recursive=True)
                                   if os.path.basename(
                        os.path.dirname(json_file)) not in self.skip_batches])
        # Split the collected JSON files into manageable chunks
        multiproc_input = [(x, json_keys) for x
                           in chunk_list(json_files, self.batch_size)]
        num_processes = cpu_count()
        log.info(f"Reading {len(json_files)} records using {num_processes} "
                 f"processes")
        # Use a multiprocessing pool to process JSON files in parallel
        with Pool(num_processes) as pool:
            res = list(tqdm(pool.map(Database._process_chunk, multiproc_input)))
        log.info(f"Inserting records into the database")
        # Insert the processed data into the database
        if table_name == self.dev_img_table:
            for item in tqdm(res, desc=f"{self.batch_size} images inserted: "):
                self._insert_dev_images(table_name, item)
        elif table_name == self.cutouts_table:
            for item in tqdm(res, desc=f"{self.batch_size} cutouts inserted: "):
                self._insert_cutouts(table_name, item)
        self._check_table(table_name)

    def _insert_one_dev_image(self, table_name: str, row: Dict[str, Any]) -> None:
        """
        Insert a single developed record into the database.

        Args:
            table_name (str): The name of the developed table.
            row (Dict[str, Any]): The record data as a dictionary.
        """
        try:
            self.cursor.execute(f"""
                            INSERT INTO {table_name} (
                                season, datetime, bbot_version,batch_id, image_id,
                                    validated, exif_meta, camera_info, 
                                    annotations, categories, version
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                row['season'], row['datetime'], row['bbot_version'],
                row['batch_id'], row['image_id'], row['validated'],
                row['exif_meta'], row['camera_info'],
                row['annotations'],
                row['categories'], row['version']
            ))
        except sqlite3.Error as e:
            log.error(
                f"{row['batch_id']}, {row['image_id']} - {e}")

    def _insert_dev_images(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """
        Insert multiple developed image records into the database.
        Uses bulk insertion if the batch size is large (over 10000), otherwise inserts one-by-one.

        Args:
            table_name (str): The developed table name.
            data (List[Dict[str, Any]]): A list of record dictionaries.
        """
        if self.batch_size < 10000:
            # Insert each record individually
            for row in tqdm(data):
                self._insert_one_dev_image(table_name, row)
            self.connection.commit()
        else:
            # Prepare a list of lists for bulk insertion
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
                # On bulk failure, rollback and insert each row individually
                self.connection.rollback()
                log.warning(
                    f"Error bulk inserting, inserting {len(bulk_data)} rows individually")
                for row in tqdm(data):
                    self._insert_one_dev_image(table_name, row)
                self.connection.commit()
        self.connection.commit()

    def _insert_one_cutout(self, table_name: str, row: Dict[str, Any]) -> None:
        """
        Insert a single cutout record into the database.

        Args:
            table_name (str): The name of the cutout table.
            row (Dict[str, Any]): The record data as a dictionary.
        """
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
                f"{row['batch_id']}, {row['image_id']}, {row['cutout_id']} - {e}")

    def _insert_cutouts(self, table_name: str, data: List[Dict[str, Any]]) -> None:
        """
        Insert multiple cutout records into the database.
        Uses bulk insertion if the batch size is large, otherwise inserts one-by-one.

        Args:
            table_name (str): The cutout table name.
            data (List[Dict[str, Any]]): A list of record dictionaries.
        """
        if self.batch_size < 10000:
            # Insert each record individually
            for row in tqdm(data):
                self._insert_one_cutout(table_name, row)
            self.connection.commit()
        else:
            # Prepare bulk data for insertion
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
                # On bulk failure, rollback and insert each row individually
                self.connection.rollback()
                log.warning(
                    f"Error bulk inserting, inserting {len(bulk_data)} rows individually")
                for row in tqdm(data):
                    self._insert_one_cutout(table_name, row)
                self.connection.commit()

    def pipeline_insert(self, batch_id, table_config):
        """
        Insert records from JSON files that match a specific batch_id.
        Processes the files in parallel and inserts each record individually.

        Args:
            batch_id (str): The batch identifier to filter JSON files.
            table_config (DictConfig): Configuration for table insertion, containing:
                - bulk_insert_paths: List of paths to search for JSON files.
                - table_name: The target table name.
                - json_keys: List of keys to serialize.
        """
        json_files = []
        # Filter JSON files based on the batch_id and table type
        for path in table_config.bulk_insert_paths:
            if len(json_files) == 0 and table_config.table_name == self.dev_img_table:
                json_files = [json_file for json_file in
                              glob.glob(path, recursive=True)
                              if os.path.basename(os.path.dirname(
                        os.path.dirname(json_file))) == batch_id]
                log.info(f"Found {len(json_files)} image metadata in {path}")
            elif len(
                    json_files) == 0 and table_config.table_name == self.cutouts_table:
                json_files = [json_file for json_file in
                              glob.glob(path, recursive=True)
                              if os.path.basename(
                        os.path.dirname(json_file)) == batch_id]
                log.info(f"Found {len(json_files)} cutout metadata in {path}")
            else:
                break
        # Chunk the list of JSON files for multiprocessing
        multiproc_input = [(x, table_config.json_keys) for x
                           in chunk_list(json_files, self.batch_size)]
        num_processes = cpu_count()
        log.info(f"Reading {len(json_files)} records using {num_processes} "
                 f"processes")
        # Process JSON chunks in parallel
        with Pool(num_processes) as pool:
            res = list(tqdm(pool.map(Database._process_chunk, multiproc_input)))
        # Insert each record individually into the database
        for item in tqdm(res, desc=f"{self.batch_size} records inserted: "):
            for row in item:
                if table_config.table_name == self.cutouts_table:
                    self._insert_one_cutout(table_config.table_name, row)
                elif table_config.table_name == self.dev_img_table:
                    self._insert_one_dev_image(table_config.table_name, row)


def main(cfg: DictConfig) -> None:
    """
    Main entry point for the database insertion pipeline.
    Creates the necessary tables and performs data insertion either via bulk_insert
    or pipeline_insert depending on the configuration.
    """
    db = Database(cfg.database)
    db.create_developed_table()
    db.create_cutouts_table()
    developed_images_cfg = cfg.database.developed_images
    cutouts_cfg = cfg.database.cutouts

    if cfg.database.bulk_insert:
        # db.bulk_insert_developed_table()
        # Bulk insert mode: process all JSON files found in the configured paths.
        db.bulk_insert(developed_images_cfg.table_name,
                       developed_images_cfg.bulk_insert_paths,
                       developed_images_cfg.json_keys)
        db.bulk_insert(cutouts_cfg.table_name,
                       cutouts_cfg.bulk_insert_paths,
                       cutouts_cfg.json_keys)
    else:
        # Pipeline insert mode: only process records matching the specified batch_id.
        db.pipeline_insert(cfg.general.batch_id,
                           developed_images_cfg)
        db.pipeline_insert(cfg.general.batch_id,
                           cutouts_cfg)
    # Log final table row counts for verification
    db._check_table(developed_images_cfg.table_name)
    db._check_table(cutouts_cfg.table_name)
