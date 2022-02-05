import csv
from dataclasses import dataclass, field

@dataclass
class Metadata:    
    """ Data class to hold metadata """

    csv_path:       str
    upload_id:      str    
    time:           str    
    location:       str    
    cloud_cover:    str    
    camera_height:  float  
    camera_lens:    float  
    pot_height:     float  
    name:           str     = "SemiField BenchBot Metadata"

    def __init__(self, csv_path):
        self.csv_path       = csv_path
        csvdict             = self.csv2dict()
        self.upload_id      = csvdict["upload_id"]
        self.time           = csvdict["time"]
        self.location       = csvdict["location"]
        self.cloud_cover    = csvdict["cloud_cover"]
        self.camera_height  = float(csvdict["camera_height"])
        self.camera_lens    = float(csvdict["camera_lens"])
        self.pot_height     = float(csvdict["pot_height"])

    def csv2dict(self):
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            result = {}
            for row in reader:
                for column, value in row.items():
                    result[column] = value
        return result  
