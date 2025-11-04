#!/bin/bash
for name in $(cat $(dirname "$0")/../splits/small-train.csv | grep -v name | awk -F ',' '{print $1}'); do
    mkdir -p $(dirname "$0")/../output/03-data_preprocess/05-small_dataset/04-raw-small/${name}

    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip \
        -O $(dirname "$0")/../output/03-data_preprocess/05-small_dataset/04-raw-small/${name}/${name}_protein.zip
    
    unzip $(dirname "$0")/../output/03-data_preprocess/05-small_dataset/04-raw-small/${name}/${name}_protein.zip \
        -d $(dirname "$0")/../output/03-data_preprocess/05-small_dataset/04-raw-small/${name}
    
    rm $(dirname "$0")/../output/03-data_preprocess/05-small_dataset/04-raw-small/${name}/${name}_protein.zip
done
