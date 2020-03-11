# Question-Driven Summarization of Answers to Consumer Health Questions
This repository contains the code to process the data and run the answer summarization systems presented in the paper Question-Driven Summarization of Answers to Consumer Health Questions
If you are interested in just downloading the data, please refer to https://doi.org/10.17605/OSF.IO/FYG46. However, if you are interested in repeating the experiments reported in the paper, clone this repository and move the data found at https://doi.org/10.17605/OSF.IO/FYG46 to the evaluation/data directory.

## Environments
To train the models and run the experiments, you will need to set up a few environments: data processing and evaluation; the BiLSTM; BART; and the Pointer-Generator. Since we are going to be processing the data first, create the following environment to install the data processing and evaluation dependencies
```
conda create -n qdriven_env python=3.7
conda activate qdriven_env
pip install -r requirements.txt
```
The requirements.txt file is found in the base directory of this repository.   
The spacy tokenizer model will be handy later on as well
```
python -m spacy download en_core_web_sm
```
And because py-rouge uses nltk, we also need nltk:
```
python
>>> import nltk
>>> nltk.download('punkt')
```
There are more details on the environments required to train each model in the following sections.


## Answer Summarization
In the models directory, there are six systems:   
Deep learning:
1.  BiLSTM
2.  Pointer Generator
3.  BART

Baselines:   
1.  LEAD-k   
2.  Random-k sentences   
3.  k ROUGE sentences   


### Runnning baselines
Running the baselines is simple and can done while the qdriven_env is active.
```
python baselines.py --dataset=chiqa
```
This will run the baseline summarization methods on the two summmarization tasks reported in the paper, as well as on the shorter passages. k (number of sentences selected by the baselines) can be changed in the script.


### Running deep learning
This code is set up to train and run inference with all models first, and then use the summarization_evaluation.py script to evaluate all results at once. This section describes the steps for training and inference.


#### Training Preprocessing
The models first have to be trained before they can be used for summarization. This requires gaining access to the BioASQ data. To do this, you have to register for an account at http://bioasq.org/participate.

Once the bioasq data is downloaded, it should be placed in the data_processing/data directory in the cloned github repository, so that the path looks like data_processing/data/BioASQ-training7b/BioASQ-training7b/training7b.json. Note that the version of BioASQ may be changed at the time of downloaded and this will have to be fixed in the code.

Once the data is in the correct place, run the following scripts:
```
python process_bioasq_data.py -p
python prepare_training_data.py -bt --bart-bioasq --bioasq-sent
```
This will prepare separate training sets for the three deep learning models. Include the ```--add-q``` option to create additional datasets with the question concatenated to the beginning of the documents, for question-driven summarization. This step will take a while to finish.

Note: Not sure if I am going to include this processing step.
Just make sure to give credence to medinfo
Then, prepare the MedInfo validation data for the Pointer-Generator and BART.
First download the .xlsx MedInfo file at https://github.com/abachaa/Medication_QA_MedInfo2019

To prepare the MedInfo validation data for the Pointer-Generator and BART. The BiLSTM does not need the MedInfo data:
```
python prepare_validation_data.py --pg --bart
```
Again, it is optional to include the --add-q option.   
Now you are ready for training and inference.


#### BART
Install the fairseq library and download BART into the bart directory in this repository. 
```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```
Then prepare an environment for BART. This also requires a few NVIDIA packages for optimized training:
```
conda create -n pytorch_env python=3.7
conda activate pytorch_env
conda install -n qdriven_env pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -n qdriven_env -c anaconda nccl
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  --global-option="--deprecated_fused_adam" ./
pip install fairseq
```
These instructions are provided in the main fairseq readme (https://github.com/pytorch/fairseq) but we have provided them here in condensed form. Note that to install apex, first make sure your GCC compiler is up-to-date.   

Once you have fairseq installed and BART downloaded to the bart directory, there are a few steps you have to take to get the bioasq in suitable format for finetuning BART.
First, run
```
bash process_bioasq_data.sh -b -f
```
This will prepare the byte-pair encodings and run the fairseq preprocessing. Once the processing is complete, you can finetune BART with
```
bash finetune_bart_bioasq.sh without_question
```
If you have been testing question-driven summarization, include with_question instead. The larger your computing cluster, the faster you will be able to train. For the experiments presented in the paper, we trained BART for two days on three V100-SXM2 GPUs with 32GB of memory each. The bash script provided here is currently configured to run with one GPU; however, the fairseq library supports multi-gpu training. 

Once you have finetuned the model, run inference on the MEDIQA-AnS dataset with
```
bash run_chiqa.sh without_question
```
Or use with_question if you have trained the appropriate model.   
For convenience, we have also included a finetuned BART model available at X. Once you have downloaded this and placed it in the models/bart/<checkpoint-for-experient> directory, you can use it to run inference.


#### BiLSTM
You will first need to set up a tensorflow2-gpu environent.
```
conda create -n tf2_env tensorflow-gpu=2.0 python=3.7
conda activate tf2_env
pip install -r requirments.txt
python -m spacy download en_core_web_sm
```
Use the requirements.txt file located in the models/bilstm directory.   
Once the environment is set up, you are ready for training. This is quite a bit easier than the previous two models.
```
train_sentence_classifier.sh
```
The training script will automatically save the checkpoint the performs that best on the validation set. The training will end after 10 epochs. The training script is configured for TensorBoard and you can monitor the loss by running tensorboard in the medsumm_bioasq_abs2summ directory.

Once the BiLSTM is trained, the following script is provided to run the model on all single document summarization tasks in MEDIQA-Ans:
```
run_chiqa.sh
```
You are now able to evaluate the BiLSTM output with the evaluation script. During inference, the run_classifier.py script will also create output files that can be used as input for inference with the Pointer-Generator or BART.k can be changed in the run_chiqa.sh script to experiment with passing top k sentences to the generative models. 


#### Pointer-Generator
Navigate to the models/pointer_generator directory and create a new environment:
```
conda create -n tf1_env python=3.6
conda activate tf1_env
wget https://files.pythonhosted.org/packages/cb/4d/c9c4da41c6d7b9a4949cb9e53c7032d7d9b7da0410f1226f7455209dd962/tensorflow_gpu-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
pip install tensorflow_gpu-1.2.0-cp36-cp36m-manylinux1_x86_64.whl 
pip install -r requirements.txt
```
The tensorflow 1.2.0 version is only availble via download from the pypi.org.   
To train the model, you will have to install cuDNN X and CUDA X. Once these are configured on your machine, you are ready for training.
The Python 3 version of the Pointer-Generator code from https://github.com/becxer/pointer-generator/ (forked from https://github.com/abisee/pointer-generator) is provided in the models/pointer_generator directory here. The code has been customized to support answer summarization data processing steps, involving changes to data.py, batcher.py, decode.py, and run_summarization.py. However, the model (in model.py) remains the same.

To use the Pointer-Generator, from the pointer_generator directory you will have to run 
```
python make_asumm_pg_vocab.py --vocab_path=bioasq_abs2summ_vocab --data_file=../../data_processing/data/bioasq_abs2summ_training_data_without_question.json
```
first, to prepare the BioASQ vocab. This is an important step, and make sure that you create the vocab WITH the [QUESTION?] tag if you are focusing on question-driven summarization. This is done simply by first creating the BioASQ data with the --add-q option.  

Then, to train, you will need to run two jobs: One to train, and the other to evaluate the checkpoints simultaneously. Run these commands independently, on two different GPU nodes:
```
bash train_medsumm.sh
bash eval_medsumm.sh
```
If you have access to a computing cluster that uses slurm, you may find it useful to use sbatch to submit these jobs.   
You will have to monitor the training of the Pointer-Generator via tensorboard and manually end the job once the loss has satisfactorily converged. The checkpoint that best performs on the MedInfo validation set will be saved to variable-name-of-experiment-directory/eval/checkpoint_best

Once it is properly trained (the MEDIQA-AnS paper reports results after 10,000 training steps), run inference on full text with the web pages with
```
run_chiqa.sh
```
The question driven option can be changed in the bash script. Note that the single pass decoding in the original Pointer-Generator code is quite slow, and it will unfortunately take approximately 45 minutes per dataset to perform inference.
Other experiments can be run configuring the script to generate summaries for the passages or multi-document datasets as well.


### Evaluation
Once the models are training and the baselines have been run on the summarization datasets you are interested in evaluating, activate the qdriven_env environment and navigate to the evluation directory. To run the evaluation script on the summarization models' predictions, you have a few options:
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

If you are interested in generating the statistics describing the collection, run 
```collection_statistics.py --tokenize``` 
in the evaluation directory. This will generate the statistics reported in the paper with more technical detail.
