from dataclasses import dataclass


@dataclass
class PipelineKeys:
    down_dev: str
    up_dev: str
    down_cut: str
    up_cut: str
    ms_lic: str