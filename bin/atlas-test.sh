#!/bin/bash
for name in $(cat $(dirname "$0")/../splits/atlas_test.csv | grep -v name | awk -F ',' '{print $1}'); do
    mkdir -p $(dirname "$0")/../output/07-testing/074-test_evaluation/a-atlas/${name}

    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip \
        -O $(dirname "$0")/../output/07-testing/074-test_evaluation/a-atlas/${name}/${name}_protein.zip
    
    unzip $(dirname "$0")/../output/07-testing/074-test_evaluation/a-atlas/${name}/${name}_protein.zip \
        -d $(dirname "$0")/../output/07-testing/074-test_evaluation/a-atlas/${name}
    
    rm $(dirname "$0")/../output/07-testing/074-test_evaluation/a-atlas/${name}/${name}_protein.zip
done

