import csv
from dataclasses import dataclass, field


@dataclass
class Metadata:
    
    """ Data class to hold metadata """

    csv_path:       str
    name:           str     = "SemiField BenchBot Metadata"
    upload_id:      str     = field(init=False, repr=True)
    time:           str     = field(init=False, repr=True)
    location:       str     = field(init=False, repr=True)
    cloud_cover:    str     = field(init=False, repr=True)
    camera_height:  float   = field(init=False, repr=True, metadata={"unit":"meters"})
    camera_lens:    float   = field(init=False, repr=True,metadata={"unit":"millimeters"})
    pot_height:     float   = field(init=False, repr=True, metadata={"unit":"meters"})
    
    def __post_init__(self):
        csvdict = self.csv2dict()
        self.time = csvdict["time"]
        self.upload_id = csvdict["upload_id"]
        self.location = csvdict["location"]
        self.cloud_cover = csvdict["cloud_cover"]
        self.camera_lens = float(csvdict["camera_lens"])
        self.camera_height = float(csvdict["camera_height"])
        self.pot_height = float(csvdict["pot_height"])
        
    
    def csv2dict(self):
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            result = {}
            for row in reader:
                for column, value in row.items():
                    result[column] = value
        return result    