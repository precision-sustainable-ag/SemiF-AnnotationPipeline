BATCHES="./blob_container/semifield-developed-images-trial/*/"
for batch in $BATCHES
do 
    IMAGEDIR="${batch%/*}"
    echo $IMAGEDIR
    python SEMIF.py data.batchdir=$IMAGEDIR general.multitask=True general.multitasks="[segment_vegetation, synthesize]"
done
