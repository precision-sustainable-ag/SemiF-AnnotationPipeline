##################################################################################
##############TUTORIAL############################################################
##################################################################################
"""
MongDB terms:

document    ->  Individual record like a single image
field       ->  Attributes of documents like _id, row, stop, 
                datetime, file_name, etc. Each document is 
                automatically given a unique "_id"
collection  ->  Documents are gathered together in collections
database    ->  Stores one or more collection of documents
"""

##################################################################################
"""
CSV to database collection
1. Converts a csv of image names and metadata into a dataframe 
2. Converts the dataframe into a dictionary
3. Inserts multiple documents into a mongodb
"""

#from mongoengine import Document, StringField, IntField, UUIDField, connect
#import pandas as pd

## Get csv records
#df = pd.read_csv("~/SemiF-AnnotationPipeline/data/sample_images.csv")
#rec = df.to_dict('records')

## Connect to DB
# connect(db="opencv2021", host="localhost", port=27017)

## Create DB class
# class Image(Document):
#     week = IntField(required=True)
#     row = IntField(required=True)
#     stop = IntField(required=True)
#     date = IntField(required=True)
#     time = IntField(required=True)
#     file_name = StringField(required=True, unique=True)
#     uuid = UUIDField(required=True)

## Inserts list of Image.objects into DB
## Uncomment line below.
# recrds = [Image(**re) for re in rec]
# Image.objects.insert(recrds, load_bulk=False)

##################################################################################
##################################################################################
"""
Load collection that already exists.
1. Connect to DB
2. Create DB schema
3. Print contents
"""

# import matplotlib.pyplot as plt
# from mongoengine import DynamicDocument, connect

# # Connect to DB
# connect(db="opencv2021", host="localhost", port=27017)

# # Create DB class
# class Image(DynamicDocument):
#     meta = {'collection': 'image'}  # Load bd that already exists

# # show examples of images
# sample_size = 3

# for img in Image.objects[0:sample_size]:
#     title = f"{img.week}, row {img.row}, stop {img.stop}"
#     img = plt.imread(f"SemiF-AnnotationPipeline/data/sample/{img.file_name}")
#     plt.title(title)
#     plt.imshow(img)
#     plt.show()

##################################################################################
##################################################################################
"""
Add field to already created collection using DynamicDocument
1. Load dict with missing field (week) and unique relatable field (file_name)
2. Create DynamicDocument DB class
3. Relate dict with DB and add field
4. Save DB
"""

# from mongoengine import connect, DynamicDocument
# import pandas as pd

# # Connect to DB
# connect(db="opencv2021", host="localhost", port=27017)
# df = pd.read_csv("~/SemiF-AnnotationPipeline/data/sample_images.csv")
# rec_wfield = df.to_dict('records')

# # Create DB class
# class Image(DynamicDocument):
#     meta = {'collection':'image'} # Load bd that already exists

# # Add new field from list of dictionaries
# for rec in rec_wfield:
#     img = Image.objects(file_name=rec["file_name"]) # returns queryset
#     if img:
#         img = img.get(file_name=rec["file_name"]) # return object from queryset
#         img.week = rec["week"]
#         img.save()

##################################################################################
##################################################################################
"""
View collection info, get all documents, and view all fields
"""

# from mongoengine import connect, DynamicDocument

# connect(db="opencv2021", host="localhost", port=27017)

# class Image(DynamicDocument):
#     meta = {
#         "collection":"image"
#     }

# dbinfo = Image.objects.explain() # DB info
# dbfields = Image.objects.all() # Gets all documents in a DB

# # Get all fields
# img = Image()
# for i in img:
#     # View fields ordered
#     print(i._fields_ordered)
#     # print all class attributes
#     attrs = vars(i)
#     print(i)

##################################################################################
##################################################################################
"""
Remove a field from collection
"""
# from mongoengine import DynamicDocument, StringField, connect

# connect(db="opencv2021", host="localhost", port=27017)

# class Image(DynamicDocument):
#     dummy_field = StringField()
#     meta = {
#         "collection": "image",
#     }

# for img in Image.objects:
#     dummy_field_str = "this is a test insertion and removal"
#     # Add dummy field to db
#     # img.dummy_field = dummy_field_str
#     # Remove dummy field from db
#     img.update(unset__dummy_field=True)
#     img.save()
