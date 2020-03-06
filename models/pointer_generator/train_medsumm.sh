#!/bin/bash
#SBATCH --output=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%x_%j.out
#SBATCH --error=/data/saveryme/asumm/models/pointer-generator/slurm_logs/slurm_%x_%j.error
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:1 
#SBATCH --mem=40g 
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00

#experiment=medsumm_train_1
# Clear tensorboard logs
#rm -r ${experiment}/*
# Train model:
# Run the original summarization model on CNN data, for testings:
#python run_medsumm.py --medsumm=False --mode=train --data_path=/data/saveryme/asumm/asumm_data/cnn-dailymail/finished_files/chunked/train_*.bin --vocab_path=/data/saveryme/asumm/asumm_data/cnn-dailymail/finished_files/vocab --exp_name=medsumm_train_0

# Train the pg model on the answer summarization task
# Start here and learn more medlineplus pages > health topic summary:
#python -u run_medsumm.py --mode=train --data_path=/data/saveryme/asumm/asumm_data/training_data/medlineplus_training_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/health_topics_vocab --exp_name=$experiment --restore_best_model=False

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
. /home/saveryme/activate_pointer_gen_env.sh

# BioASQ abstracts > snippets with no END tags at end of summaries
# With question:
if [ $QDRIVEN == "with_question" ]; then
    echo $QDRIVEN
    experiment=bioasq_abs2summ_with_question
    rm -r ${experiment}/*
    python -u run_medsumm.py --mode=train --data_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_training_data_with_question.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment
fi

# And without question:
if [ $QDRIVEN == "without_question" ]; then
    echo $QDRIVEN
    experiment=bioasq_abs2summ_without_question
    rm -r ${experiment}/*
    python -u run_medsumm.py --mode=train --data_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_training_data_without_question.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment
fi

# The issue with training the next two models is that there is only ~2000 examples

# BioASQ abstracts > single answer with no tags at end of summaries and ARTICLE SEP tag between abstracts
#experiment=bioasq_abs2ans
#rm -r ${experiment}/*
#python -u run_medsumm.py --mode=train --data_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2ans_training_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2ans_vocab --exp_name=$experiment

# BioASQ snippets > single answer with no END tags at end of summaries and with ARTICLE SEP tag between snippet
#experiment=bioasq_snips2ans
#rm -r ${experiment}/*
#python -u run_medsumm.py --mode=train --data_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_snips2ans_training_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_snips2ans_vocab --exp_name=$experiment

# Iterate through datasets and train respective models
# Haven't tried to run this code yet
#declare -a arr=("bioasq_abs2ans" "bioasq_abs2summ" "bioasq_snips2ans")
#for experiment in
#  do
#  echo $experiment
#  rm -r ${experiment}/*
#    python -u run_medsumm.py --mode=train --data_path=/data/saveryme/asumm/asumm_data/training_data/${experiment}_training_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/{experiment}_vocab --exp_name=$experiment
