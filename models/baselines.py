"""
Script for baseline approaches:
1. Take 10 random sentences
2. Take the topk 10 sentences with highest rouge score relative to the question
3. Pick the first 10 sentences

To run
python baselines.py --dataset=chiqa
"""

import json
import numpy as np
import random
import requests
import argparse

import rouge
import spacy


def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--dataset",
                        dest="dataset",
                        help="Dataset to run baselines on. Only current option is MEDIQA-AnS.")
    return parser


def calculate_sentence_level_rouge(question, doc_sen, evaluator):
    """
    For each pair of sentences, calculate rouge score with py-rouge
    """
    rouge_score = evaluator.get_scores(doc_sen, question)['rouge-l']['f']
    return rouge_score


def pick_k_best_rouge_sentences(k, questions, documents, summaries):
    """
    Pick the k sentences that have the highest rouge scores when compared to the question.
    """
    # Initiate rouge evaluator
    evaluator = rouge.Rouge(metrics=['rouge-l'],
                          max_n=3,
                          limit_length=False,
                          length_limit_type='words',
                          apply_avg=False,
                          apply_best=True,
                          alpha=1,
                          weight_factor=1.2,
                          stemming=False)

    pred_dict = {
        'question': [],
        'ref_summary': [],
        'gen_summary': []
        }
    for q, doc, summ, in zip(questions, documents, summaries):
        # Sentencize abstract
        rouge_scores = []
        for sentence in doc:
            rouge_score = calculate_sentence_level_rouge(q, sentence, evaluator)
            rouge_scores.append(rouge_score)
        if len(doc) < k:
            top_k_rouge_scores = np.argsort(rouge_scores)
        else:
            top_k_rouge_scores = np.argsort(rouge_scores)[-k:]
        top_k_sentences = " ".join([doc[i] for i in top_k_rouge_scores])
        summ = summ.replace("<s>", "")
        summ = summ.replace("</s>", "")
        pred_dict['question'].append(q)
        pred_dict['ref_summary'].append(summ)
        pred_dict['gen_summary'].append(top_k_sentences)

    return pred_dict


def pick_first_k_sentences(k, questions, documents, summaries):
    """
    Pick the first k sentences to use as summaries
    """
    pred_dict = {
        'question': [],
        'ref_summary': [],
        'gen_summary': []
        }
    for q, doc, summ, in zip(questions, documents, summaries):
        if len(doc) < k:
            first_k_sentences = doc
        else:
            first_k_sentences = doc[0:k]
        first_k_sentences = " ".join(first_k_sentences)
        summ = summ.replace("<s>", "")
        summ = summ.replace("</s>", "")
        pred_dict['question'].append(q)
        pred_dict['ref_summary'].append(summ)
        pred_dict['gen_summary'].append(first_k_sentences)

    return pred_dict


def pick_k_random_sentences(k, questions, documents, summaries):
    """
    Pick k random sentences from the articles to use as summaries
    """
    pred_dict = {
        'question': [],
        'ref_summary': [],
        'gen_summary': []
        }
    random.seed(13)
    for q, doc, summ, in zip(questions, documents, summaries):
        if len(doc) < k:
            random_sentences = " ".join(doc)
        else:
            random_sentences = random.sample(doc, k)
            random_sentences = " ".join(random_sentences)
        summ = summ.replace("<s>", "")
        summ = summ.replace("</s>", "")
        pred_dict['question'].append(q)
        pred_dict['ref_summary'].append(summ)
        pred_dict['gen_summary'].append(random_sentences)

    return pred_dict


def load_dataset(path):
    """
    Load the evaluation set
    """
    with open(path, "r", encoding="utf-8") as f:
        asumm_data = json.load(f)

    summaries = []
    questions = []
    documents = []
    nlp = spacy.load('en_core_web_sm')
    # Split sentences 
    cnt = 0
    for q_id in asumm_data:
        questions.append(asumm_data[q_id]['question'])
        tokenized_art = nlp(asumm_data[q_id]['articles'])
        summaries.append(asumm_data[q_id]['summary'])
        article_sentences = [s.text.strip() for s in tokenized_art.sents]
        documents.append(article_sentences[0:])
    return questions, documents, summaries


def save_baseline(baseline, filename):
    """
    Save baseline in format for rouge evaluation
    """
    with open("../evaluation/data/baselines/chiqa_eval/baseline_{}.json".format(filename), "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=4)


def run_baselines():
    """
    Generate the random baseline and the best rouge baseline
    """
    # Load the MEDIQA-AnS datasets
    datasets = [
        ("../data_processing/data/page2answer_single_abstractive_summ.json", "p2a-single-abs"),
        ("../data_processing/data/page2answer_single_extractive_summ.json", "p2a-single-ext"),
        ("../data_processing/data/section2answer_single_abstractive_summ.json", "s2a-single-abs"),
        ("../data_processing/data/section2answer_single_extractive_summ.json", "s2a-single-ext"),
    ]

    for data in datasets:
        task = data[1]
        print("Running baselines on {}".format(task))
        # k can be determined from averages or medians of summary types of reference summaries. Alternatively, just use Lead-3 baseline.
        # Optional to use different k for extractive and abstractive summaries, as the manual summaries of the two types have different average lengths
        if task == "p2a-single-abs":
            k = 3
        if task == "p2a-single-ext":
            k = 3
        if task == "s2a-single-abs":
            k = 3
        if task == "s2a-single-ext":
            k = 3
        questions, documents, summaries = load_dataset(data[0])
        k_sentences = pick_k_random_sentences(k, questions, documents, summaries)
        first_k_sentences = pick_first_k_sentences(k, questions, documents, summaries)
        k_best_rouge = pick_k_best_rouge_sentences(k, questions, documents, summaries)

        save_baseline(k_sentences, filename="random_sentences_k_{}_{}_{}".format(k, args.dataset, task))
        save_baseline(first_k_sentences, filename="first_sentences_k_{}_{}_{}".format(k, args.dataset, task))
        save_baseline(k_best_rouge, filename="best_rouge_k_{}_{}_{}".format(k, args.dataset, task))


if __name__ == "__main__":
    args = get_args().parse_args()
    run_baselines()
