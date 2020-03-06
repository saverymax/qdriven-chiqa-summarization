#!/bin/bash
#SBATCH --output=/data/saveryme/asumm/models/fairseq/examples/bart/slurm_logs/slurm_%j.out
#SBATCH --error=/data/saveryme/asumm/models/fairseq/examples/bart/slurm_logs/slurm_%j.error
#SBATCH --job-name=tr-bart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:3
#SBATCH --mem=20g
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=16
BART_PATH=bart/bart.large/model.pt
checkpoint_path=checkpoints_bioasq_$SLURM_JOB_ID
q_string=without_question
asumm_data=/data/saveryme/asumm/asumm_data/training_data/bart/${q_string}/bart-bin

CUDA_VISIBLE_DEVICES=0,1,2 python train.py ${asumm_data} \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --truncate-source \
    --task translation \
    --source-lang source --target-lang target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --ddp-backend=no_c10d \
    --save-dir=${checkpoint_path} \
    --keep-last-epochs=2 \
    --find-unused-parameters;
    #--max-tokens-valid $MAX_TOKENS \
