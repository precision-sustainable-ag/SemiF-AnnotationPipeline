from dataclasses import dataclass


@dataclass
class PipelineKeys:
    account_url: str
    down_dev: str
    up_dev: str
    down_cut: str
    up_cut: str
    down_upload: str
    up_upload: str

    ms_lic: str
