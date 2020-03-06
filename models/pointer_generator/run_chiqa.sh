#!/bin/bash
#SBATCH --output=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%j.out
#SBATCH --error=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%j.error
#SBATCH --job-name=chiq_eval
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100x:1 
#SBATCH --mem=10g 
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00

sent_class_dir=/data/saveryme/asumm/models/sentence_classifier/medsumm_bioasq_abs2summ/dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary
k=10
for q in without_question with_question
do
    experiment=bioasq_abs2summ_${q}
    if [ ${q} == "with_question" ]; then
        q_driven=True
    else
        q_driven=False
    fi
    for summ_task in page2answer section2answer
    do
        for summ_type in singleAbstractive singleExtractive
        do
            # Use sentence classifier output and raw input
            for use_sent_class in y n
            do
                if [ ${use_sent_class} == "y" ]; then
                    sent_class_file=predictions_chiq_${summ_task}_${summ_type}_dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_topk${k}.json
                    input_data=${sent_class_dir}/${sent_class_file}
                    predict_file=pointergen_chiqa_bioasq_abs2summ_${q}_sent_class_${k}_${summ_task}_${summ_type}.json
                else
                    data=chiqa_${summ_task}_${summ_type}Sums_test_data.json
                    input_data=/data/saveryme/asumm/asumm_data/chiqa_test_data/${data}
                    predict_file=pointergen_chiqa_bioasq_abs2summ_${q}_${summ_task}_${summ_type}.json
                fi
                echo ${q} ${q_driven}
                echo $input_data
                echo $predict_file
                python run_medsumm.py \
                    --mode=decode \
                    --data_path=${input_data} \
                    --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab \
                    --exp_name=$experiment \
                    --single_pass=True \
                    --eval_type=medsumm \
                    --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}/${predict_file} \
                    --tag_sentences=True \
                    --question_driven=${q_driven}

            done
        done
    done
done
