#!/bin/bash
# Process bioasq data generated in the prepare_training_data.py script
# -b for byte pair encoding
# -f for the rest of the fairseq processing 
# Paths for loading training data including the question is specified by the q_string variable, currently hardcoded
# Keep in mind that while the q_string will NOT change the name of any files in the bart-bin directory.

echo "Args: $@"
for q_string in without_question with_question
do 
    echo ${q_string}
    asumm_data=/data/saveryme/asumm/asumm_data/training_data/bart/${q_string}
    echo asumm_data
    if [ "$1" == "-b" ]
    then
        echo "Running byte-pair encoding"
        for SPLIT in train val
        do
          for LANG in source target
          do
            python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json \
            --vocab-bpe vocab.bpe \
            --inputs ${asumm_data}/bart.${SPLIT}_${q_string}.$LANG \
            --outputs ${asumm_data}/bart.${SPLIT}_${q_string}.bpe.$LANG \
            --workers 60 \
            --keep-empty;
          done
        done
    fi

    if [ "$2" == "-f" ]
    then
        echo "Running fairseq processing"
        fairseq-preprocess \
          --source-lang "source" \
          --target-lang "target" \
          --trainpref ${asumm_data}/bart.train_${q_string}.bpe \
          --validpref ${asumm_data}/bart.val_${q_string}.bpe \
          --destdir ${asumm_data}/bart-bin \
          --workers 60 \
          --srcdict dict.txt \
          --tgtdict dict.txt;
    fi
done
