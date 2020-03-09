#!/bin/bash

validation_data=../../data_processing/data/medinfo_section2answer_validation_data_${1}.json
# Bioasq abs2summ evaluating on medinfo page > section with question
if [ $1 == "with_question" ]; then
    echo $1
    experiment=bioasq_abs2summ_with_question/
    python -u run_medsumm.py --mode=eval --data_path=$validation_data --vocab_path=bioasq_abs2summ_vocab --exp_name=$experiment
fi

# And same thing but without question
if [ $1 == "without_question" ]; then
    echo $1
    experiment=bioasq_abs2summ_without_question/
    python -u run_medsumm.py --mode=eval --data_path=$validation_data --vocab_path=bioasq_abs2summ_vocab --exp_name=$experiment
fi
