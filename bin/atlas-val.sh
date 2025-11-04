for name in $(cat $(dirname "$0")/../output/05-data_preprocess/54-code/547-small_dataset/csv/small-val.csv | grep -v name | awk -F ',' '{print $1}'); do
    mkdir -p $(dirname "$0")/../output/05-data_preprocess/54-code/542-raw/${name}

    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_protein.zip \
        -O $(dirname "$0")/../output/05-data_preprocess/54-code/542-raw/${name}/${name}_protein.zip
    
    unzip $(dirname "$0")/../output/05-data_preprocess/54-code/542-raw/${name}/${name}_protein.zip \
        -d $(dirname "$0")/../output/05-data_preprocess/54-code/542-raw/${name}
    
    rm $(dirname "$0")/../output/05-data_preprocess/54-code/542-raw/${name}/${name}_protein.zip
done
