#!/bin/bash
# Process bioasq data generated in the prepare_training_data.py script
# -b as first argument for byte pair encoding
# -f as second argument for the rest of the fairseq processing 
# Specify with_question or without_question as third argument

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

echo "Args: $@"
asumm_data=bart_config/${3}
echo "Data dir: $asumm_data"
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
        --inputs ${asumm_data}/bart.${SPLIT}_${3}.$LANG \
        --outputs ${asumm_data}/bart.${SPLIT}_${3}.bpe.$LANG \
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
      --trainpref ${asumm_data}/bart.train_${3}.bpe \
      --validpref ${asumm_data}/bart.val_${3}.bpe \
      --destdir ${asumm_data}/bart-bin \
      --workers 60 \
      --srcdict dict.txt \
      --tgtdict dict.txt;
fi
