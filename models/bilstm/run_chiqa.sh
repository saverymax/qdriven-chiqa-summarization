#!/bin/bash
exp_name=medsumm_bioasq_abs2summ
tensorboard_log=dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary

for summ_task in page2answer section2answer
do
    for summ_type in single_abstractive single_extractive
    do
        for k in 3
        do
            # Optional to specify different k, for selecting top k sentences. If you are interested in using the output of the sentence classifier as input for a generative model,
            # you may want to increase k
            echo "k:" $k
            data=../../data_processing/data/${summ_task}_${summ_type}_summ.json
            echo $data
            prediction_file=predictions_chiqa_${summ_task}_${summ_type}_dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_topk${k}.json
            eval_file=../../evaluation/data/sentence_classifier/chiqa_eval/sent_class_chiqa_${summ_task}_${summ_type}_dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_topk${k}.json
            python run_classifier.py \
                --exp_name=${exp_name} \
                --mode=infer \
                --data_path=$data \
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
                --eval_file=$eval_file \
                --tag_sentences=False \
                --top_k_sent=${k}
        done
    done
done
