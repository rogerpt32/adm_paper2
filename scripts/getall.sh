#!/bin/bash
cd ../src
for filename in ../imgs/*; do
    python3 extract.py $filename
done
for filename in ../data/*; do
    name=${filename#"../data/"}
    name=${name%".csv"}
    for ((i=2; i<=6; i++)); do
        python3 clustering.py $name $i
    done
done
