BATCHES="./data/semifield-developed-images/*/"

for batch in $BATCHES
do 
    IMAGEDIR="$(basename $batch)"
    AUTOSFMDIR=${batch%%/}/"autosfm"
    DEVELOPEDIMAGES=${batch%%/}/"images"
    MASKS=${batch%%/}/"masks"
    DETECTIONS=${batch%%/}/"plant-detections"
    # echo $IMAGEDIR
    # echo $AUTOSFMDIR
    
    ### Check if a directory does not exist ###
    if [[ ! -d "$DEVELOPEDIMAGES" && ! -d "$MASKS" ]]
    then
        echo "Both images and masks for $IMAGEDIR are being downloaded from blob storage..."
        python scripts/my_download_blob_parallel.py --batch_id $IMAGEDIR --images --masks
    
    fi
    if [[ -d "$DEVELOPEDIMAGES" && ! -d "$MASKS" ]]
    then
        echo "Only masks for $IMAGEDIR are being downloaded from blob storage..."
        python scripts/my_download_blob_parallel.py --batch_id $IMAGEDIR --masks
    fi
    if [ ! -d "$DETECTIONS" ]
    then
        echo "Only Detections for $IMAGEDIR are being downloaded from blob storage..."
        python scripts/my_download_blob_parallel.py --batch_id $IMAGEDIR --detections
    fi
    if [ ! -d "$AUTOSFMDIR" ] 
    then
        export agisoft_LICENSE="/home/weedsci/metashape-pro/metashape.lic"
        echo "Running AutoSfM for $IMAGEDIR"
        python SEMIF.py \
        general.batch_id=$IMAGEDIR \
        autosfm.metashape_key="E2ONG-2JDX3-147TH-5EFU7-832BJ" \
        general.task=auto_sfm
        
    else
        echo "AutoSfM already run for $IMAGEDIR, skipping..."
    fi
done
echo "Script run to completion"