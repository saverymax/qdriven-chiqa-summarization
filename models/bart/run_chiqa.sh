#!/bin/bash
for q in with_question without_question
do
    model_config=/data/saveryme/asumm/asumm_data/training_data/bart/${q}/bart-bin
    if [ ${q} == "with_question" ]; then
        model_path=checkpoints_bioasq_47384726/
        echo "Using question driven model" $model_path
    else
        model_path=checkpoints_bioasq_47386235
        echo "Using model without questions" $model_path
    fi
    # With sentence classifier output:
    for summ_task in page2answer section2answer
    do
        for summ_type in singleAbstractive singleExtractive
        do
            sent_class_file=predictions_chiq_${summ_task}_${summ_type}_dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_topk${k}.json
            input_file=${sent_class_dir}/${sent_class_file}
            prediction_file=bart_chiqa_${q}_${summ_task}_${summ_type}_sent_class_binary_topk${k}.json
            prediction_path=/data/saveryme/asumm/asumm_data/predictions/bart_predictions/${prediction_file}
            echo $input_file
            echo $prediction_path
            python run_inference_medsumm.py \
                --input_file=$input_file \
                --question_driven=${q} \
                --prediction_file=$prediction_path \
                --model_path=$model_path \
                --model_config=$model_config
        done
    done
    # And with raw CHiQA input:
    for summ_task in page2answer section2answer
    do
        for summ_type in singleAbstractive singleExtractive
        do
            data=chiqa_${summ_task}_${summ_type}Sums_test_data.json
            input_file=/data/saveryme/asumm/asumm_data/chiqa_test_data/${data}
            prediction_file=bart_chiqa_${q}_${summ_task}_${summ_type}.json
            prediction_path=/data/saveryme/asumm/asumm_data/predictions/bart_predictions/${prediction_file}
            echo $input_file
            echo $prediction_path
            python run_inference_medsumm.py \
                --input_file=$input_file \
                --question_driven=${q} \
                --prediction_file=$prediction_path \
                --model_path=$model_path \
                --model_config=$model_config
        done
    done
done
