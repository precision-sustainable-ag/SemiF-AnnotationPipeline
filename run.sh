BATCHES="./data/trial/*/"
for batch in $BATCHES
do 
    IMAGEDIR="${batch%/*}"
    echo $IMAGEDIR
    python SEMIF.py general.batchdir=$IMAGEDIR general.multitask=True
done
