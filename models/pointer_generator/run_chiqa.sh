#!/bin/bash
#SBATCH --output=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%j.out
#SBATCH --error=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%j.error
#SBATCH --job-name=chiq_eval
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100x:1 
#SBATCH --mem=10g 
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00

# Can also include the with_question string in for loop
for q in without_question 
do
    experiment=bioasq_abs2summ_${q}
    if [ ${q} == "with_question" ]; then
        q_driven=True
    else
        q_driven=False
    fi
    for summ_task in page2answer section2answer
    do
        for summ_type in single_abstractive single_extractive
        do
            data=${summ_task}_${summ_type}_summ.json
            input_data=../../data_processing/data/${data}
            predict_file=pointergen_chiqa_bioasq_abs2summ_${q}_${summ_task}_${summ_type}.json
            echo ${q} ${q_driven}
            echo $input_data
            echo $predict_file
            python run_medsumm.py \
                --mode=decode \
                --data_path=${input_data} \
                --vocab_path=./bioasq_abs2summ_vocab \
                --exp_name=$experiment \
                --single_pass=True \
                --eval_type=medsumm \
                --generated_data_file=../../evaluation/data/pointer_generator/chiqa_eval/${predict_file} \
                --tag_sentences=True \
                --question_driven=${q_driven}
        done
    done
done
