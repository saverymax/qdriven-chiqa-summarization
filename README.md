# Question-Driven Summarization of Answers to Consumer Health Questions
This repository contains the code to process the data and run the answer summarization systems presented in the paper Question-Driven Summarization of Answers to Consumer Health Questions

## Data Processing
Download MEDIQA-AnS from https://doi.org/10.17605/OSF.IO/FYG46 to the evaluation/data directory in this repository.

## Answer Summarization
In the models directory, there are six systems:
Deep learning:
1.  BiLSTM
2.  Pointer Generator
3.  BART

Baselines:
4.  LEAD-k   
5.  Random-k sentences   
6.  k ROUGE sentences   

### Running deep learning
This code is set up to train and run inference with all models first, and then use the summarization_evaluation.py script to evaluate all results at once. This section describes the steps for training and inference.

#### Training Preprocessing
The models first have to be trained before they can be used for summarization. This requires gaining access to the BioASQ data. To do this, you have to register for an account at http://bioasq.org/participate.

Once the bioasq data is downloaded, it should be placed in the data_processing/data directory in the cloned github repository, so that the path looks like data_processing/data/BioASQ-training7b/BioASQ-training7b/training7b.json. Note that the version of BioASQ may be changed at the time of downloaded and this will have to be fixed in the code.

Once the data is in the correct place, run the following scripts:
```
python process_bioasq_data.py -d -p
python prepare_training_data.py -bt --bart-bioasq --bioasq-sent
```
This will download the pubmed articles used as reference in the bioasq collection, and prepare separate training sets for the three deep learning models. Include the ```--add-q``` option to create additional datasets with the question concatenated to the beginning of the documents, for question-driven summarization.

Note: Not sure if I am going to include this processing step.
Then, prepare the MedInfo validation data for the Pointer-Generator and BART.
First download the .xlsx MedInfo file at https://github.com/abachaa/Medication_QA_MedInfo2019

To prepare the MedInfo validation data for the Pointer-Generator and BART:
```
python prepare_validation_data.py -t
python prepare_validation_data.py
```
It is optional to include the --add-q option.
Now you are ready for training and inference.

#### Pointer-Generator
The Python 3 version of the Pointer-Generator code from https://github.com/becxer/pointer-generator/ (forked from https://github.com/abisee/pointer-generator) is provided in the models/pointer_generator directory here. The code has been customized to support answer summarization data processing steps, involving changes to data.py, batcher.py, decode.py, and run_summarization.py. However, the model (in model.py) remains the same.

To use the Pointer-Generator, you will have to run ```make_asumm_pg_vocab.py``` first, to prepare the BioASQ vocab. Then, to train, you will need to run two jobs: One to train, and the other to evaluate the checkpoints simultaneously. Run these commands independently, on two different GPU nodes:
```
bash train_medsumm.sh
bash eval_medsumm.sh
```
If you have access to a computing cluster that uses slurm, you may find it useful to use sbatch to submit these jobs.
You will have to monitor the training of the Pointer-Generator via tensorboard and manually end the job once the loss has satisfactorily converged. The checkpoint that best performs on the MedInfo validation set will be saved to variable-name-of-experiment-directory/eval/checkpoint_best

Once it is properly trained, run inference with
```
run_chiqa.sh
```
The single pass decoding in the original Pointer-Generator code is quite slow, and it will unfortunately take approximately 45 minutes per dataset to perform inference.


#### BART
Install the fairseq library and download BART into the bart directory in this repository. We have provided a simple script to download BART.
```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
pip install fairseq
```
Once you have fairseq installed and BART downloaded to the bart directory, there are a few steps you have to take to get the bioasq in suitable format for finetuning BART.
First, run
```
process_bioasq_data.sh -b -f
```
This will prepare the byte-pair encodings and run the fairseq preprocessing. Once the processing is complete, you can finetune BART with
```
finetune_bart_bioasq.sh
```
The larger your computing cluster, the faster you will be able to train. For the experiments presented in the paper, we trained BART for two days on three V100-SXM2 GPUs with 32GB of memory each.

Once you have finetuned the model, run inference on the MEDIQA-AnS dataset with
```
run_chiqa.sh
```
It may take a while to run through all the datasets set up in the bash script.

#### BiLSTM
This is quite a bit easier to train than the previous two models. Just run
```
train_sentence_classifier.sh
```
The training script will automatically save the checkpoint the performs that best on the validation set. The training will end after 10 epochs. The training script is configured for TensorBoard and you can monitor the loss by running tensorboard in the medsumm_bioasq_abs2summ directory.

Once the BiLSTM is trained, to run the model on all single document summarization tasks in MEDIQA-Ans, run
```
run_chiqa.sh
```
You are now able to evaluate the BiLSTM output with the evaluation script.

### Runnning baselines
To run the baseline systems, all you have to do is
```
python baselines.py --dataset=chiqa
```
This will run the baseline summarization methods on the two summmarization tasks reported in the paper, as well as on the shorter passages.

### Evaluation
Once the models are training and the baselines have been run on the summarization datasets you are interested in evaluating, navigate to the evluation directory. To run the evaluation script on the summarization models' predictions, you have a few options:
For comparing all models on extractive and abstractive summaries of web pages:
```
python summarization_evaluation.py --dataset=chiqa --bleu --evaluate-models
```
Or to run two versions of BART (question-driven approach and without questions) run
```
python summarization_evaluation.py --dataset=chiqa --bleu --q-driven
```
The same question-driven test can be applied to the Pointer-Generator as well, if you have trained the appropriate model with the correctly formatted question-driven dataset.

Other options, such as saving scores per summary to file, or calculating Wilcoxon p-values, are described in the script.


If you are interested in generating the statistics describing the collection, run the ```collection_statistics.py``` script in the evaluation directory. This will generate the statistics reported in the paper with more technical detail.
