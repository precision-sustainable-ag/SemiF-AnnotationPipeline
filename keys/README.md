
# Keys

Create a file in this `keys` directory and name it `pipeline_keys.yaml`. Populate the file using the below format. The names must be exactly the same as what is below.
Populate the necessary fields using azure SAS keys for each blob. Be sure to give the appropriate permissions. 

:warning: **Always use short term SAS keys to avoid giving longterm access to the public in the case that they are accidently exposed.**

Place the metasahpe license key in the the last field, `metashape.lic`

```yaml
SAS:
  developed:
    download:
    upload: 
  
  cutouts:
    download: 
    upload: 
  
  weedsimagerepo:
    download: 

metashape:
  lic: 
```