BATCHES="./data/semifield-developed-images-trial/*/"

for batch in $BATCHES
do 
    IMAGEDIR="$(basename $batch)"
    # IMAGEDIR=$batch
    echo $IMAGEDIR
    python SEMIF.py \
    general.batch_id=$IMAGEDIR \
    general.task=segment_vegetation \
    autosfm.metashape_key=asd
    # general.multitask=True \
    # general.multitasks="[segment_vegetation]" \
    
done