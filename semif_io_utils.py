import io

import yaml

# data = 1

# Write YAML file
# with io.open('data.yaml', 'w', encoding='utf8') as outfile:
#     yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

def read_batch_metadata(path):
    # Read YAML file
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        return data_loaded
