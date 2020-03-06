#!/bin/bash
# Perform inference with trained model:
# The model training on CNN
#python run_summarization.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/cnn-dailymail/finished_files/chunked/train_*.bin --vocab_path=/data/saveryme/asumm/asumm_data/cnn-dailymail/finished_files/vocab --exp_name=medsumm_train_0

# Health topics summarization model:
#experiment=medsumm_train_1
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/medinfo_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/health_topics_vocab --exp_name=$experiment

#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/medinfo_page2section_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/health_topics_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/medsumm_train_1/pointer_gen_generated_medinfo.json

# Model trained on bioasq abstracts to snippets, including question:
experiment=bioasq_abs2summ/
# Run on medinfo section > answer
python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_section2answer_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_abs2summ_generated_medinfo_sec2ans.json

# Run on medinfo webpage > answer
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_page2answer_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_abs2summ_generated_medinfo_page2ans.json

# Run on medinfo webpage > section
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_page2section_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_abs2summ_generated_medinfo_page2sec.json

# Using bioasq abtracts to answer model
#experiment=bioasq_abs2ans/
# Run on medinfo section > answer
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_section2answer_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2ans_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_abs2summ_generated_medinfo_sec2ans.json

# Run on medinfo webpage > answer
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_page2answer_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2ans_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_abs2summ_generated_medinfo_page2ans.json

# Run on medinfo webpage > section
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_page2section_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2ans_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_abs2summ_generated_medinfo_page2sec.json

# Using bioasq snip2ans model
# section > answer 
#experiment=bioasq_snips2ans/
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_section2answer_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_snips2ans_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_snips2ans_generated_medinfo_sec2ans.json

# Run on medinfo webpage > answer
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_page2answer_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_snips2ans_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_snips2ans_generated_medinfo_sec2ans.json

# Run on medinfo webpage > section
#python run_medsumm.py --mode=decode --data_path=/data/saveryme/asumm/asumm_data/validation_data/medinfo_page2section_validation_data.json --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_snips2ans_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}pointergen_bioasq_snips2ans_generated_medinfo_sec2ans.json

# Run on sentences extracted by sentence classifier > answer
#experiment=bioasq_abs2summ
#prediction_file=predictions_medinfo_page2answer_validation_data_dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_exact_match_topk5.json
#tensorboard_log=dropout_5_sent_200_tok_50_val_20_d_256_l2_reg_binary_combined_snip
#extracted_sentences=/data/saveryme/asumm/models/sentence_classifier/medsumm_bioasq_abs2summ/${tensorboard_log}/${prediction_file}
#generated_summary=pointergen_bioasq_abs2summ_generated_medinfo_extracted-sent2answer_sent_200_binary_exact_match_topk5.json
#python run_medsumm.py --mode=decode --data_path=${extracted_sentences} --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}/${generated_summary}


# Run on summaries generated by open summarizer
#experiment=bioasq_abs2summ
#prediction_file=osumm_extracted_medinfo_validation_summ_for_pg.json
#extracted_sentences=/data/saveryme/asumm/asumm_data/osumm_extracted/${prediction_file}
#generated_summary=pointergen_bioasq_abs2summ_generated_medinfo_extracted-osumm.json
#python run_medsumm.py --mode=decode --data_path=${extracted_sentences} --vocab_path=/data/saveryme/asumm/asumm_data/training_data/bioasq_abs2summ_vocab --exp_name=$experiment --single_pass=True --eval_type=medsumm --generated_data_file=/data/saveryme/asumm/models/pointer-generator/${experiment}/${generated_summary}
