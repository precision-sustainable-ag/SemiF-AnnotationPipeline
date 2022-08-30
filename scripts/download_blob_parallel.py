# https://www.quickprogrammingtips.com/azure/how-to-download-blobs-from-azure-storage-using-python.html
# https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?tabs=environment-variable-windows#download-blobs

import os
from multiprocessing.pool import Pool, ThreadPool

from azure.storage.blob import BlobServiceClient
from omegaconf import DictConfig

# IMPORTANT: Replace connection string with your storage account connection string
# Usually starts with DefaultEndpointsProtocol=https;...
MY_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=weedsimagerepo;AccountKey=ob1kY+RNLFILbh661w7NKECdFRtlDwlpwSwpZbcRlKOXR49Qp6JLRpUcken5TUzbUVqPr/pG5XEJ+ASt/2ap7Q==;EndpointSuffix=core.windows.net"

# Replace with blob container name
MY_BLOB_CONTAINER = "semifield-developed-images"

# Replace with the local folder where you want downloaded files to be stored
LOCAL_BLOB_PATH = "data/semifield-developed-images"
 
class AzureBlobFileDownloader:
  def __init__(self, batch_id):
    print("Intializing AzureBlobFileDownloader")
    self.batch_id = batch_id
    # Initialize the connection to Azure storage account
    self.blob_service_client =  BlobServiceClient.from_connection_string(MY_CONNECTION_STRING)
    self.my_container = self.blob_service_client.get_container_client(MY_BLOB_CONTAINER)
 
  def download_all_blobs_in_container(self):
    # get a list of blobs
    my_blobs = self.my_container.list_blobs(name_starts_with=self.batch_id)
    my_blobs = iter([x for x in my_blobs if not x.name.endswith(".pp3")])
    result = self.run(my_blobs)
    print(result)
 
  def run(self,blobs):
    # Download 10 files at a time!
    cpu_count = os.cpu_count() - 1
    with Pool(processes=cpu_count) as p:
      p.map(self.save_blob_locally, blobs)
      p.join()
      p.close()
    # with ThreadPool(processes=int(10)) as pool:
    #  return pool.map(self.save_blob_locally, blobs)
 
  def save_blob_locally(self,blob):
    file_name = blob.name
    print(file_name)
    bytes = self.my_container.get_blob_client(blob).download_blob().readall()

    # Get full path to the file
    download_file_path = os.path.join(LOCAL_BLOB_PATH, file_name)
    # for nested blobs, create local path as well!
    os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
 
    with open(download_file_path, "wb") as file:
      file.write(bytes)
    return file_name

def main(cfg: DictConfig) -> None:
  batch_id = cfg.general.batch_id
  # Initialize class and upload files
  azure_blob_file_downloader = AzureBlobFileDownloader(batch_id)
  azure_blob_file_downloader.download_all_blobs_in_container()
