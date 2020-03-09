#!/bin/bash
#SBATCH --output=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%j.out
#SBATCH --error=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%j.error
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:1 
#SBATCH --mem=40g 
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00

training_data=../../data_processing/data/bioasq_abs2summ_training_data_${1}.json 
# With question:
if [ $1 == "with_question" ]; then
    echo $1
    mkdir bioasq_abs2summ_with_question
    experiment=bioasq_abs2summ_with_question
    rm -r ${experiment}/*
    python -u run_medsumm.py --mode=train --data_path=$training_data --vocab_path=./bioasq_abs2summ_vocab --exp_name=$experiment
fi

# And without question:
if [ $1 == "without_question" ]; then
    echo $1
    mkdir bioasq_abs2summ_without_question
    experiment=bioasq_abs2summ_without_question
    rm -r ${experiment}/*
    python -u run_medsumm.py --mode=train --data_path=$training_data --vocab_path=./bioasq_abs2summ_vocab --exp_name=$experiment
fi
