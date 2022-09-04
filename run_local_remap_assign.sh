BATCHES="./data/semifield-developed-images/*/"

for batch in $BATCHES
do 
    IMAGEDIR="$(basename $batch)"
    METADATADIR=${batch%%/}/"metadata"
    PLANTDETECTIONS=${batch%%/}/"plant-detections"
    DETECTIONS=${batch%%/}/"autosfm/detections.csv"
    
    ### Check if a directory does not exist ###
    if [[ ! -f "$DETECTIONS" && ! -d "$PLANTDETECTIONS" ]]
    then
        echo "Running detections for $IMAGEDIR ..."
        python SEMIF.py \
        general.batch_id=$IMAGEDIR \
        autosfm.metashape_key="E2ONG-2JDX3-147TH-5EFU7-832BJ" \
        general.task=localize_plants    
    fi

    if [[ -d "$PLANTDETECTIONS" && ! -d "$METADATADIR" ]]
    then
        echo "Concating detections that have already been performed for $IMAGEDIR ..."
        python SEMIF.py \
        general.batch_id=$IMAGEDIR \
        autosfm.metashape_key="E2ONG-2JDX3-147TH-5EFU7-832BJ" \
        general.task=localize_plants \
        detect.concat_detections=True
    fi

    if [ ! -d "$METADATADIR" ]
    then
        echo "Remapping detections and assigning species for $IMAGEDIR ..."
        python SEMIF.py \
        general.batch_id=$IMAGEDIR \
        autosfm.metashape_key="E2ONG-2JDX3-147TH-5EFU7-832BJ" \
        general.task=remap_labels

        python SEMIF.py \
        general.batch_id=$IMAGEDIR \
        autosfm.metashape_key="E2ONG-2JDX3-147TH-5EFU7-832BJ" \
        general.task=assign_species
    else
        echo "No remapping for $IMAGEDIR"
    fi
done
echo "Script run to completion"