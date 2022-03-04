from mongoengine import (Document, IntField, ListField, StringField, UUIDField,
                         connect)

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


class Cutout(Document):
    orig_img_file_name = StringField(required=True)
    cutout_uuid = UUIDField(required=True, unique=True)
    cutout_fname = StringField(required=True, unique=True)
    cutout_num = IntField(required=True)
    cutout_contour = ListField(required=True)
    meta = {'allow_inheritance': True}
