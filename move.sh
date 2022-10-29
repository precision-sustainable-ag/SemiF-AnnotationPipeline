# BATCHES="./data/semifield-developed-images/*/"
BATCHES="/media/hdd1/mkutu/wir/semifield-developed-images/*/"

HDDDIR="/media/hdd1/mkutu/wir/semifield-developed-images"
DSTDIR="/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-developed-images"
for batch in $BATCHES
do 
    
    BATCHDIR="$(basename $batch)"
    
    HDD_PLANTDETECTIONS=${batch%%/}/"plant-detections"
    HDD_MASKSDIR=${batch%%/}/"masks"
    HDD_DEVIMAGESDIR=${batch%%/}/"images"
    HDD_AUTOSFMDIR=${batch%%/}/"autosfm"
    HDD_METAMASKS=${batch%%/}/"meta_masks"
    HDD_METADATA=${batch%%/}/"metadata"
    HDD_BATCHMETA=${batch%%/}/"$BATCHDIR.json"

    MAIN_PLANTDETECTIONS=$DSTDIR/$BATCHDIR/"plant-detections"
    MAIN_MASKSDIR=$DSTDIR/$BATCHDIR/"masks"
    MAIN_DEVIMAGESDIR=$DSTDIR/$BATCHDIR/"images"
    MAIN_AUTOSFMDIR=$DSTDIR/$BATCHDIR/"autosfm"
    MAIN_METAMASKS=$DSTDIR/$BATCHDIR/"meta_masks"
    MAIN_METADATA=$DSTDIR/$BATCHDIR/"metadata"
    MAIN_BATCHMETA=$DSTDIR/$BATCHDIR/"$BATCHDIR.json"
    
    MAIN_BATCHDIR=$DSTDIR/$BATCHDIR

    ## Check if everything is there
    if [[ ! -d "$HDD_AUTOSFMDIR" || ! -d "$HDD_PLANTDETECTIONS" || ! -d "$HDD_MASKSDIR" || ! -d "$HDD_DEVIMAGESDIR" ]]
    then
        echo "Skipping $BATCHDIR because of missing main folders..."
        continue
    fi

    ## Copy from HDD to Main
    if [ -d "$HDD_AUTOSFMDIR" ]
    then
        mkdir -p $MAIN_AUTOSFMDIR
        cp -r $HDD_AUTOSFMDIR $MAIN_BATCHDIR
        # echo "$HDD_AUTOSFMDIR ..."
    fi
    

    if [ -d "$HDD_PLANTDETECTIONS" ]
    then
        mkdir -p $MAIN_PLANTDETECTIONS
        cp -r $HDD_PLANTDETECTIONS $MAIN_BATCHDIR
        # echo "$HDD_PLANTDETECTIONS ..."
    fi
    

    if [ -d "$HDD_MASKSDIR" ]
    then
        mkdir -p $MAIN_MASKSDIR
        cp -r $HDD_MASKSDIR $MAIN_BATCHDIR
        # echo "$HDD_MASKSDIR ..." 
    fi
    
    if [ -d "$HDD_DEVIMAGESDIR" ]
    then
        mkdir -p $MAIN_DEVIMAGESDIR
        cp -r $HDD_DEVIMAGESDIR $MAIN_BATCHDIR
        # echo "$HDD_DEVIMAGESDIR ..."      
    fi

    ## DO SOMETHING HERE
    bash runrun.sh

    ## Move things back to HDD
    mkdir -p $HDD_METAMASKS
    cp -r $MAIN_METAMASKS $batch
    
    mkdir -p $HDD_METADATA
    cp -r $MAIN_METADATA $batch

    # mkdir -p $HDD_BATCHMETA
    cp  $MAIN_BATCHMETA $batch
    
    ##Remove from Main
    rm -r $MAIN_PLANTDETECTIONS
    rm -r $MAIN_MASKSDIR
    rm -r $MAIN_DEVIMAGESDIR
    rm -r $MAIN_METAMASKS
    rm -r $MAIN_METADATA
    rm -r $MAIN_BATCHMETA
    RM -R $MAIN_AUTOSFMDIR
    
    rm -r ~/.local/share/Trash/*
    
    echo "Done for $BATCHDIR..."


done