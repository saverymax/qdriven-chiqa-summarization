#!/bin/bash
# Include with_question or without_question when calling script
model_config=bart_config/${1}/bart-bin
model_path=checkpoints_bioasq_$1
for summ_task in page2answer section2answer
do
    for summ_type in single_abstractive single_extractive
    do
        data=${summ_task}_${summ_type}_summ.json
        input_file=../../data_processing/data/${data}
        prediction_file=bart_chiqa_${1}_${summ_task}_${summ_type}.json
        prediction_path=../../evaluation/data/bart/chiqa_eval/${prediction_file}
        echo $input_file
        echo $prediction_path
        python run_inference_medsumm.py \
            --input_file=$input_file \
            --question_driven=$1 \
            --prediction_file=$prediction_path \
            --model_path=$model_path \
            --model_config=$model_config
    done
done
