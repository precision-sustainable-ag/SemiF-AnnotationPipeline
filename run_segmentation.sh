BATCHES="./data/semifield-developed-images/*/"

for batch in $BATCHES
do 
    IMAGEDIR="$(basename $batch)"
    
    ### Check if a directory does not exist ###
    echo "Running segmentation for $IMAGEDIR ..."
    python SEMIF.py \
    general.batch_id=$IMAGEDIR \
    autosfm.metashape_key="E2ONG-2JDX3-147TH-5EFU7-832BJ" \
    general.task=segment_vegetation
done