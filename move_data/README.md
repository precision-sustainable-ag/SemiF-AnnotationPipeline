# MoveData

Either downloads or uploads data from blob container.

## Download Processed batches from blob storage

1. Make sure you have an up-to-date SAS key
2. Run `manual_download.py` to get a list of downloadable batches from the blob. the script checks to make sure each batch folder contains:
      - `images`
      - `meta_masks`
      - `metadata`
      - `logs`
      - batch metadata `.json` files
    
    A list of batches is placed in `.batchlogs/batch_download.txt`  
3. Adjust `.batchlogs/batch_download.txt` list to accomodate memory availability.
    Only download a certain number of batches at a time to reduce the amount of memory you're occupying. 
4. Run `blob2nfs.sh <path to batch_download.txt> <output path>`
5. Finally, move the downloaded batches to NFS storage by running:
    ```bash
    mv <path to batch folder> /mnt/research-projects/s/screberg/longterm_storage/semifield-developed-images/
    ```

