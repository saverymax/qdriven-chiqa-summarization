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

###Running deep learning
####Training
The models first have to be trained before they can be used for summarization. This requires gaining access to the BioASQ data. To do this, you have to register for an account at http://bioasq.org/participate.

A script in provided in the data_processing directory to process the downloaded BioASQ data into a format suitable for training the models. 
to download the pubmed articles for each snippet run
python process_bioasq.py -d
then to process the questions, answers, and snippets, run:
python process_bioasq.py -p
or to process all of this but join the snippets that are taken from the same
abstract but are listed separately in the file:
python process_bioasq.py -pj


#####Pointer-Generator
#####BART
#####BiLSTM


###Runnning baselines
To run the baseline systems, all you have to do is
```
python baselines.py --dataset=chiqa
```
This will run the baseline summarization methods on the two summmarization tasks reported in the paper, as well as on the shorter passages.

###Evaluation
