#!/bin/bash

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
. /home/saveryme/activate_pointer_gen_env.sh

# Bioasq abs2summ evaluating on medinfo page > section with question
if [ $QDRIVEN == "with_question" ]; then
    echo $QDRIVEN
    validation_data=medinfo_section2answer_validation_data_with_question.json
    experiment=bioasq_abs2summ_with_question/
    python -u run_medsumm.py --mode=eval --data_path=/data/saveryme/asumm/asumm_data/validation_data/$validation_data --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment
fi

# And same thing but without question
if [ $QDRIVEN == "without_question" ]; then
    echo $QDRIVEN
    validation_data=medinfo_section2answer_validation_data_without_question.json
    experiment=bioasq_abs2summ_without_question/
    python -u run_medsumm.py --mode=eval --data_path=/data/saveryme/asumm/asumm_data/validation_data/$validation_data --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment
fi
