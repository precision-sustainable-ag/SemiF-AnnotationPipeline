from move_data.utils.data import PipelineKeys
import yaml


def read_keys(keypath):
    with open(keypath, 'r') as file:
        pipe_keys = yaml.safe_load(file)
        sas = pipe_keys['SAS']
        up_cut = sas['cutouts']['upload']
        down_cut = sas['cutouts']['download']

        up_dev = sas['developed']['upload']
        down_dev = sas['developed']['download']
        keys = PipelineKeys(down_dev=down_dev,
                            up_dev=up_dev,
                            down_cut=down_cut,
                            up_cut=up_cut,
                            ms_lic=pipe_keys['metashape']['lic'])
    return keys