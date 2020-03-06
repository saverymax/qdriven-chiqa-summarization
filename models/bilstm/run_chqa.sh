#!/bin/bash
vocab_file=bioasq_abs2summ_vocab
exp_name=medsumm_bioasq_abs2summ
tensorboard_log=dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary

for summ_task in page2answer section2answer
do
    for summ_type in singleAbstractive singleExtractive
    do
        for k in 3 10
        do
            echo $k
            data=chiqa_${summ_task}_${summ_type}Sums_test_data.json
            echo $data
            prediction_file=predictions_chiq_${summ_task}_${summ_type}_dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_topk${k}.json
            eval_file=sent_class_chiqa_${summ_task}_${summ_type}_dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_topk${k}.json
            python run_classifier.py \
                --vocab_path=/data/saveryme/asumm/asumm_data/training_data/${vocab_file} \
                --exp_name=${exp_name} \
                --mode=infer \
                --data_path=/data/saveryme/asumm/asumm_data/chiqa_test_data/${data} \
                --dataset=chiqa \
                --summary_type=${summ_task}_${summ_type} \
                --train_tokenizer=False \
                --tokenizer_path=./${exp_name}/tokenizer \
                --model_path=${exp_name}/${tensorboard_log}/bioasq_abs2summ_sent_class_model.h5 \
                --batch_size=32 \
                --max_sentences=200 \
                --max_tok_sent=50 \
                --max_tok_q=50 \
                --dropout=.5 \
                --hidden_dim=256 \
                --binary_model=True \
                --prediction_file=${exp_name}/${tensorboard_log}/${prediction_file} \
                --eval_file=${exp_name}/${tensorboard_log}/${eval_file} \
                --tag_sentences=False \
                --top_k_sent=${k}
        done
    done
done
