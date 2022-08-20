BATCHES="./data/semifield-developed-images/*/"

for batch in $BATCHES
do 
    IMAGEDIR="$(basename $batch)"
    AUTOSFMDIR=${batch%%/}/"metadata"

    # echo $IMAGEDIR
    # echo $AUTOSFMDIR
    
    ### Check if a directory does not exist ###
    if [ ! -d "$AUTOSFMDIR" ] 
    then
        export agisoft_LICENSE="/home/weedsci/matt/SemiF-AnnotationPipeline/autosfm/volumes/metashape/metashape-pro/metashape.lic"
        echo "Running remap_labels for $IMAGEDIR"
        python SEMIF.py \
        general.batch_id=$IMAGEDIR \
        autosfm.metashape_key=asd \
        general.task=assign_species
        
    else
        echo "AutoSfM already run for $IMAGEDIR, skipping..."
    fi

done
echo "Script run to completion"