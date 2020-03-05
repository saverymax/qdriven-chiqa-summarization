# Question-Driven Summarization of Answers to Consumer Health Questions
This repository contains the code to process the data and run the answer summarization systems presented in the paper Question-Driven Summarization of Answers to Consumer Health Questions

## Data Processing


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
#### Training
The models first have to be trained before they can be used for summarization. This requires gaining access to the BioASQ data. To do this, you have to register for an account at http://bioasq.org/participate.

Once the bioasq data is downloaded, it should be placed in the data_processing/data directory in the cloned github repository, so that the path looks like data_processing/data/BioASQ-training7b/BioASQ-training7b/training7b.json. Note that the version of BioASQ may be changed at the time of downloaded and this will have to be fixed in the code.   
Once the data is in the correct place, run the scripts:
```
python process_bioasq_data.py -d -p
python prepare_training_data.py -bt --bart-bioasq --bioasq-sent
```
This will convert the bioasq data into an easier to use format, download the pubmed articles used as reference in the bioasq collection, and then prepare data for the three deep learning models. Include the --add-q option to create additional datasets with the question concatenated to the beginning of the documents, for question-driven summarization.   
Then, prepare the MedInfo validation data for the Pointer-Generator and BART. This requires two commands:
```
python prepare_validation_data.py -t
python prepare_validation_data.py
```
As before, it is optional to include the --add-q option.
Now you are ready for training

##### Pointer-Generator
##### BART
##### BiLSTM


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
```
Or to run two versions of BART (question-driven approach and without questions) run
```
```
