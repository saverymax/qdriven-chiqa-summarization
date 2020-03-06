#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

#for SPLIT in train val
#do
#  for LANG in source target
#  do
#    python -m examples.roberta.multiprocessing_bpe_encoder \
#    --encoder-json encoder.json \
#    --vocab-bpe vocab.bpe \
#    --inputs "cnn_dm_finished_files/$SPLIT.$LANG" \
#    --outputs "cnn_dm_finished_files/$SPLIT.bpe.$LANG" \
#    --workers 60 \
#    --keep-empty;
#  done
#done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "cnn_dm_finished_files/train.bpe" \
  --validpref "cnn_dm_finished_files/val.bpe" \
  --destdir "cnn_dm-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
