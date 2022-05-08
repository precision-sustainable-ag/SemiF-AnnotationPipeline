from dataclasses import asdict
from getpass import getpass

from pymongo import MongoClient


# Connect to Your MongoDB Instance
# Create a class for your connection code so that it can be reused.
class Connect(object):

    @staticmethod
    def get_connection():
        password = "tamuweedsci2022"  # getpass()
        return MongoClient(
            f"mongodb://superuser:{password}@localhost:27017/admin?")


def to_db(db, collection, data):
    """Turns a single dictionary into a document and inserts it into the database

    Args:
        db (pymongo.database.Database): connection to mongodb of choice
        collection (str): mongo collection
        data (dict): dict and contents of wanted document
    """
    # Inserts dictionaries into mongodb
    data_doc = asdict(data)
    getattr(db, collection).insert_one(data_doc)


def from_db(db, collection):
    """Get all documents from mongodb collection and returns a list of documents

    Args:
        db (pymongo.database.Database): connection to mongodb of choice
        collection (str): mongo collection

    Returns:
        docs (list): list of documents
    """
    collection = db[collection]
    cursor = collection.find({})
    return list(cursor)


# def get_individual_synth_collection(cfg: DictConfig) -> None:
#     """ IDK where else to put this. TODO move this somewhere else???"""
#     db = getattr(Connect.get_connection(), cfg.general.db)
#     collection = "Cutouts"
#     docs = from_db(db, collection)
#     print(docs)
