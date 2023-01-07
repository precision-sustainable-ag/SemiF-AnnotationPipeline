# https://medialab.github.io/iwanthue/
# random list of 100 colors
# hue between 0 and 360
# c between 17.72 and 100
# luminosity between 30.77 and 95.28
# Avoids use of 255 rgb value
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

VERSION = 1.2
NOTES = {
    "1":
    "USDA symbols are used as species index key.",
    "2":
    "When difference between USDA and EPPO codes, scientific_names, or authority occured, USDA information was used.",
    "3":
    "Purple nutsedge (CYRO) is used by may also be yellow nutsedge (CYES).",
    "4":
    "North Carolina (MD), Texas (TX), Maryland (MD)",
    "5":
    "Brassicas class contains 4 species - Brassica napus, Brassica rapa, Brassica juncea, Brassica hirta"
}

# csv = "class_list.csv"
# df = pd.read_csv(csv, index_col="species_id")
# df = df.where(df.notnull(), None)
# df["rgb"] = df.rgb.map(lambda x: x.lstrip('"').rstrip("'"))
# df["rgb"] = df.rgb.str.replace("'", "", regex=True)
# df["rgb"] = df.rgb.str.replace("[", "", regex=False)
# df["rgb"] = df.rgb.str.replace("]", "", regex=False)
# df["rgb"] = df.rgb.str.split(", ").apply(lambda x: [int(i) for i in x])
# df = df.to_dict("index")
# spec_info = dict()
# spec_info["species"] = df
# spec_info["Notes"] = NOTES
# spec_info["Version"] = VERSION


# json_object = json.dumps(spec_info, indent=4)
# with open("data/semifield-utils/species_info.json", "w") as outfile:
# outfile.write(json_object)
def make_custom_sort(orders):
    orders = [{k: -i
               for (i, k) in enumerate(reversed(order), 1)}
              for order in orders]

    def process(stuff):
        if isinstance(stuff, dict):
            l = [(k, process(v)) for (k, v) in stuff.items()]
            keys = set(stuff)
            for order in orders:
                if keys.issuperset(order):
                    return OrderedDict(
                        sorted(l, key=lambda x: order.get(x[0], 0)))
            return OrderedDict(sorted(l))
        if isinstance(stuff, list):
            return [process(x) for x in stuff]
        return stuff

    return process


with open("data/semifield-utils/species_information/species_info_og.json"
          ) as outfile:
    data = json.load(outfile)
spec = data["species"]
for usda_sym in spec.keys():
    info = spec[usda_sym]["scientific_name"].split(" ")
    genus, species = info[0], info[1] if len(info) > 1 else info[0]
    spec[usda_sym]["genus"] = genus
    spec[usda_sym]["species"] = species
    # sort_order = ['site', 'A1', 'A5', 'A10']
    sort_order = [
        "class_id", "USDA_symbol", "EPPO", "group", "class", "subclass",
        "order", "family", "genus", "species", "common_name",
        "scientific_name", "authority", "growth_habit", "duration",
        "collection_location", "category", "collection_timing", "link", "note",
        "hex", "rgb"
    ]
    custom_sort = make_custom_sort([sort_order])
    # allclskeys_ordered = [
    #     OrderedDict(
    #         sorted(item.items(), key=lambda item: sort_order.index(item[0])))
    #     for item in spec[usda_sym]
    # ]

    result = custom_sort(spec[usda_sym])

    spec[usda_sym] = result
    popped = spec[usda_sym].pop("scientific_name")
data["species"] = spec
with open('data/semifield-utils/species_information/species_info.json',
          'w') as f:
    json.dump(data, f, indent=4)
    # outfile.write(json_object)
class_ids = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100
]

usda_symbols = [
    'background', 'AMPA', 'AMAR2', 'SEOB4', 'XAST', 'DISA', 'ELIN3', 'URPL2',
    'CYRO', 'AMTU', 'ECCR', 'ECCO2', 'URTE2', 'BASC5', 'HEAN3', 'PAHY', 'SOHA',
    'GLMA4', 'AMHY', 'CHAL7', 'PADI', 'DAST', 'ABTH', 'SEPU8', 'SEFA',
    'ERCA20', 'ZEA', 'plant', 'colorchecker', 'VIVI', 'PISA6', 'TRIN3',
    'TRPR2', 'BRASS2', 'RASA2', 'SECE', 'TRITI2', 'TRAE', 'AVSA', 'HORDE',
    'AVST2'
]

hex = [
    "#000000", "#1d686e", "#e452f1", "#60e54e", "#903be2", "#9ee33f",
    "#5d4ce7", "#dbe93c", "#bd36d6", "#3fac38", "#912bba", "#57eb93",
    "#e342c6", "#8fe175", "#5a42c1", "#e3ca3b", "#6364e2", "#a0b837",
    "#b360e8", "#6ba33c", "#ab2a9f", "#cfea7c", "#8e42b0", "#40be7b",
    "#e636a0", "#5aedc3", "#e83c2b", "#55ece3", "#e73b58", "#9ee3a0",
    "#617ff7", "#eea626", "#3967c5", "#e86928", "#5e97e6", "#ba3a1c",
    "#5dcfe9", "#ea427e", "#44975e", "#e172d8", "#407018", "#bd7ae4",
    "#aa952b", "#7350af", "#d6a846", "#968de9", "#d88633", "#34649f",
    "#e7d37e", "#9d4999", "#7b852d", "#b5317f", "#6fc196", "#b73361",
    "#50bbaf", "#bb353f", "#87d9dd", "#9e3442", "#a6e2cc", "#e067af",
    "#436f30", "#e2a1e5", "#6a6115", "#a376c2", "#aebd70", "#675395",
    "#d9e2ad", "#8f3d76", "#8ea975", "#e57fa7", "#436e42", "#e87159",
    "#44a7d9", "#a15921", "#9ac6ed", "#a5482f", "#3e98a4", "#e07179",
    "#306b51", "#b16fa0", "#8f6a21", "#8080c0", "#b8965b", "#4481ab",
    "#db8e6b", "#336b85", "#dab28e", "#586188", "#8c895d", "#c3b5e9",
    "#605e32", "#e6adc2", "#33675e", "#9c4d5f", "#58937e", "#7d5371",
    "#836542", "#7a9cc2", "#945848", "#9e85ac", "#bf828a"
]

rgb = [[0, 0, 0], [29, 104, 110], [228, 82, 241], [96, 229, 78],
       [144, 59, 226], [158, 227, 63], [93, 76, 231], [219, 233, 60],
       [189, 54, 214], [63, 172, 56], [145, 43, 186], [87, 235, 147],
       [227, 66, 198], [143, 225, 117], [90, 66, 193], [227, 202, 59],
       [99, 100, 226], [160, 184, 55], [179, 96, 232], [107, 163, 60],
       [171, 42, 159], [207, 234, 124], [142, 66, 176], [64, 190, 123],
       [230, 54, 160], [90, 237, 195], [232, 60, 43], [85, 236, 227],
       [231, 59, 88], [158, 227, 160], [97, 127, 247], [238, 166, 38],
       [57, 103, 197], [232, 105, 40], [94, 151, 230], [186, 58, 28],
       [93, 207, 233], [234, 66, 126], [68, 151, 94], [225, 114, 216],
       [64, 112, 24], [189, 122, 228], [170, 149, 43], [115, 80, 175],
       [214, 168, 70], [150, 141, 233], [216, 134, 51], [52, 100, 159],
       [231, 211, 126], [157, 73, 153], [123, 133, 45], [181, 49, 127],
       [111, 193, 150], [183, 51, 97], [80, 187, 175], [187, 53, 63],
       [135, 217, 221], [158, 52, 66], [166, 226, 204], [224, 103, 175],
       [67, 111, 48], [226, 161, 229], [106, 97, 21], [163, 118, 194],
       [174, 189, 112], [103, 83, 149], [217, 226, 173], [143, 61, 118],
       [142, 169, 117], [229, 127, 167], [67, 110, 66], [232, 113, 89],
       [68, 167, 217], [161, 89, 33], [154, 198, 237], [165, 72, 47],
       [62, 152, 164], [224, 113, 121], [48, 107, 81], [177, 111, 160],
       [143, 106, 33], [128, 128, 192], [184, 150, 91], [68, 129, 171],
       [219, 142, 107], [51, 107, 133], [218, 178, 142], [88, 97, 136],
       [140, 137, 93], [195, 181, 233], [96, 94, 50], [230, 173, 194],
       [51, 103, 94], [156, 77, 95], [88, 147, 126], [125, 83, 113],
       [131, 101, 66], [122, 156, 194], [148, 88, 72], [158, 133, 172],
       [191, 130, 138]]
