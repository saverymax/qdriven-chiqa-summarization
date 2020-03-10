vocab_file=bioasq_abs2summ_vocab
exp_name=medsumm_bioasq_abs2summ
tensorboard_log=dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary
mkdir $exp_name
mkdir ${exp_name}/${tensorboard_log}
training_data=../../data_processing/data/bioasq_abs2summ_binary_sent_classification_training.json
max_sent=200
binary_model=True
# Note that for first run the sub-word tokenizer has to be trained first. For all subsequent runs, can be left false.
train_tokenizer=False

python run_classifier.py \
    --vocab_path=../../data_processing/data${vocab_file} \
    --exp_name=${exp_name} \
    --tensorboard_log=${exp_name}/${tensorboard_log} \
    --mode=train \
    --data_path=$training_data \
    --train_tokenizer=$train_tokenizer \
    --tokenizer_path=./${exp_name}/tokenizer \
    --model_image_path=./${exp_name}/model_graph.png \
    --model_path=bioasq_abs2summ_sent_class_model.h5 \
    --batch_size=32 \
    --max_sentences=$max_sent \
    --max_tok_sent=50 \
    --max_tok_q=50 \
    --dropout=.5 \
    --hidden_dim=256 \
    --binary_model=$binary_model
