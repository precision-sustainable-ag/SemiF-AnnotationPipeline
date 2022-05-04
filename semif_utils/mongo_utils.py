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
