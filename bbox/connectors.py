import os
import pandas as pd

class SfMComponents:

    def __init__(self, base_path, 
                 camera_reference_file="camera_reference.csv", 
                 fov_reference_file="fov.csv", 
                 gcp_reference_file="gcp_reference.csv"):
        
        self.base_path = base_path
        self.camera_reference_file = camera_reference_file
        self.fov_reference_file = fov_reference_file
        self.gcp_reference_file = gcp_reference_file

        self._camera_reference = None
        self._fov_reference = None
        self._gcp_reference = None

    @property
    def camera_reference(self):
        if self._camera_reference is None:
            self._camera_reference = pd.read_csv(os.path.join(self.base_path, self.camera_reference_file))
        return self._camera_reference

    @property
    def fov_reference(self):
        if self._fov_reference is None:
            self._fov_reference = pd.read_csv(os.path.join(self.base_path, self.fov_reference_file))
        return self._fov_reference

    @property
    def gcp_reference(self):
        if self._gcp_reference is None:
            self._gcp_reference = pd.read_csv(os.path.join(self.base_path, self.gcp_reference_file))
        return self._gcp_reference
