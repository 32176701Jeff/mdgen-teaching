#!/bin/bash
for name in $(cat $(dirname "$0")/../splits/atlas_test.csv | grep -v name | awk -F ',' '{print $1}'); do
    mkdir -p /mnt/hdd/jeff/dataset/output/mdgen-#1/testset/${name}

    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip \
        -O /mnt/hdd/jeff/dataset/output/mdgen-#1/testset/${name}/${name}_protein.zip
    
    unzip /mnt/hdd/jeff/dataset/output/mdgen-#1/testset/${name}/${name}_protein.zip \
        -d /mnt/hdd/jeff/dataset/output/mdgen-#1/testset/${name}
    
    rm /mnt/hdd/jeff/dataset/output/mdgen-#1/testset/${name}/${name}_protein.zip
done

