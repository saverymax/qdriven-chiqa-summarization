# Question-Driven Summarization of Answers to Consumer Health Questions
This repository contains the code to process the data and run the answer summarization systems presented in the paper Question-Driven Summarization of Answers to Consumer Health Questions
If you are interested in just downloading the data, please refer to https://doi.org/10.17605/OSF.IO/FYG46. However, if you are interested in repeating the experiments reported in the paper, clone this repository and move the data found at https://doi.org/10.17605/OSF.IO/FYG46 to the evaluation/data directory.

## Environments
All instructions provided here have been tested in a Linux operating system. To train the models and run the experiments, you will need to set up a few environments with anaconda: data processing and evaluation; the BiLSTM; BART; and the Pointer-Generator. Since we are going to be processing the data first, create the following environment to install the data processing and evaluation dependencies: 
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
There are more details regarding the environments required to train each model in the following sections.


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
Running the baselines is simple and can be done while the qdriven_env environment is active.
```
python baselines.py --dataset=chiqa
```
This will run the baseline summarization methods on the two summmarization tasks reported in the paper, as well as on the shorter passages. k (number of sentences selected by the baselines) can be changed in the script.


### Running deep learning
The following code is organized to train and run inference with all models first, and then use the summarization_evaluation.py script to evaluate all results at once. This section describes the steps for training and inference.


#### Training Preprocessing
First prepare the validation data for the Pointer-Generator and BART:
```
python prepare_validation_data.py --pg --bart
```
It is optional to include the --add-q option if you are interested in training models question-driven summarization.   

To create the training data, the BioASQ data for training first has to be acquired. To do so, you have to register for an account at http://bioasq.org/participate.

In the participants area of the website, you can find a list of all the datasets previously choosed for the BioASQ challenge. Download the 7b version of the task. Once the BioASQ data has been downloaded and unizpped, it should be placed in the data_processing/data directory in the cloned github repository, so that the path relative to the data_processing directory looks like ```data_processing/data/BioASQ-training7b/BioASQ-training7b/training7b.json```. Note that we used version 7b of BioASQ for training and testing. You are welcome to experiment with 8b or newer but will have to fix the paths in the code.

Once the data is in the correct place, run the following scripts:
```
python process_bioasq.py -p
python prepare_training_data.py -bt --bart-bioasq --bioasq-sent
```
This will prepare separate training sets for the three deep learning models. Include the ```--add-q``` option to create additional datasets with the question concatenated to the beginning of the documents, for question-driven summarization. This step will take a while to finish. Once it is done, you are ready for training and inference.


#### BiLSTM (sentence classification)
You will first need to set up a tensorflow2-gpu environent and install the dependencies for the model:
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


#### BART
Download BART into the models/bart directory in this repository. 
```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```
Navigate to the models/bart directory and prepare an environment for BART. This also requires a few NVIDIA packages for optimized training.
```
conda create -n pytorch_env python=3.7 --file requirements.txt
conda activate pytorch_env
conda install -n pytorch_env pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -n pytorch_env -c anaconda nccl
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  --global-option="--deprecated_fused_adam" ./
```
These instructions are provided in the main fairseq readme (https://github.com/pytorch/fairseq) but we have provided them here in condensed form. Note that to install apex, first make sure your GCC compiler is up-to-date.   
Once these dependencies have been installed, you are ready to install fairseq. This requires installing an editable version of a earlier commit of the repository. Navigate to back to the models/bart directory of this repo and run:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout 43cf9c977b8470ec493cc32d248fdcd9a984a9f6
pip install --editable .
```
Because the fairseq repository is contains research projects under continuous development, the previous commit checked-out here should can be used to recreate the results. Using a current version of fairseq may require more troubleshooting. Once you have fairseq installed and BART downloaded to the bart directory, there are a few steps you have to take to get the bioasq in suitable format for finetuning BART.
First
```
bash process_bioasq_data.sh -b -f without_question
```
This will prepare the byte-pair encodings and run the fairseq preprocessing. with_question will prepare data for question-driven summarization, without_question for plain summaization. Once the processing is complete, you can finetune BART with
```
bash finetune_bart_bioasq.sh without_question
```
If you have been testing question-driven summarization, include with_question instead. The larger your computing cluster, the faster you will be able to train. For the experiments presented in the paper, we trained BART for two days on three V100-SXM2 GPUs with 32GB of memory each. The bash script provided here is currently configured to run with one GPU; however, the fairseq library supports multi-gpu training. 

Once you have finetuned the model, run inference on the MEDIQA-AnS dataset with
```
bash run_chiqa.sh without_question
```
Or use with_question if you have trained the appropriate model.   


#### Pointer-Generator
Navigate to the models/pointer_generator directory and create a new environment:
```
conda create -n tf1_env python=3.6
conda activate tf1_env
wget https://files.pythonhosted.org/packages/cb/4d/c9c4da41c6d7b9a4949cb9e53c7032d7d9b7da0410f1226f7455209dd962/tensorflow_gpu-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
pip install tensorflow_gpu-1.2.0-cp36-cp36m-manylinux1_x86_64.whl 
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
The tensorflow 1.2.0 version is only availble via download from pypi.org, hence the use of ```wget``` first. To train the model, you will have to install cuDNN X and CUDA X. Once these are configured on your machine, you are ready for training.   
The Python 3 version of the Pointer-Generator code from https://github.com/becxer/pointer-generator/ (forked from https://github.com/abisee/pointer-generator) is provided in the models/pointer_generator directory here. The code has been customized to support answer summarization data processing steps, involving changes to data.py, batcher.py, decode.py, and run_summarization.py. However, the model (in model.py) remains the same.

To use the Pointer-Generator, from the pointer_generator directory you will have to run 
```
python make_asumm_pg_vocab.py --vocab_path=bioasq_abs2summ_vocab --data_file=../../data_processing/data/bioasq_abs2summ_training_data_without_question.json
```
first, to prepare the BioASQ vocab. If you are focusing on question-driven summarization, provide that dataset instead.

Then, to train, you will need to run two jobs: One to train, and the other to evaluate the checkpoints simultaneously. Run these commands independently, on two different GPUs:
```
bash train_medsumm.sh without_question
bash eval_medsumm.sh without_question
```
If you have access to a computing cluster that uses slurm, you may find it useful to use sbatch to submit these jobs.   
You will have to monitor the training of the Pointer-Generator via tensorboard and manually end the job once the loss has satisfactorily converged. The checkpoint that best performs on the MedInfo validation set will be saved to variable-name-of-experiment-directory/eval/checkpoint_best

Once it is properly trained (the MEDIQA-AnS paper reports results after 10,000 training steps), run inference on full text with the web pages with
```
run_chiqa.sh
```
The question driven option can be changed in the bash script. Note that the single pass decoding in the original Pointer-Generator code is quite slow, and it will unfortunately take approximately 45 minutes per dataset to perform inference.
Other experiments can be run if you have configured the bash script to generate summaries for the passages or multi-document datasets as well.


### Evaluation
Once the models are training and the baselines have been run on the summarization datasets you are interested in evaluating, activate the qdriven_env environment again and navigate to the evaluation directory. To run the evaluation script on the summarization models' predictions, you have a few options:
For comparing all models on extractive and abstractive summaries of web pages:
```
python summarization_evaluation.py --dataset=chiqa --bleu --evaluate-models
```
Or to run two versions of BART (question-driven approach and without questions, if you have trained both) run
```
python summarization_evaluation.py --dataset=chiqa --bleu --q-driven
```
The same question-driven test can be applied to the Pointer-Generator as well, if you have trained the appropriate model with the correctly formatted question-driven dataset.

More details about the options, such as saving scores per summary to file, or calculating Wilcoxon p-values, are described in the script.

If you are interested in generating the statistics describing the collection, run 
```collection_statistics.py --tokenize``` 
in the evaluation directory. This will generate the statistics reported in the paper with more technical detail.


That's it! Thank you for using this code, and please contact us if you find any issues with the repository or have questions about summarization. If you publish work related to this project, please cite
```
```
