from mongoengine import Document, IntField, StringField, UUIDField, connect

CUTOUT_DICT = {"cutout_fname": None, "cutout_uuid": None, "contours": None}

connect(db="opencv2021", host="localhost", port=27017)


class Image(Document):
    file_name = StringField(required=True, unique=True)
    uuid = UUIDField(required=True)
    date = IntField(required=True)
    time = IntField(required=True)
    week = IntField()
    row = IntField()
    stop = IntField()
    meta = {'allow_inheritance': True}
