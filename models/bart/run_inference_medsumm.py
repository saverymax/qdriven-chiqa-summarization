"""
Script modified from fairseq example CNN-dm script, for performing 
summarization inference on input articles.
"""

import argparse
import json

from tqdm import tqdm
import torch
from fairseq.models.bart import BARTModel


def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--prediction_file",
                        dest="prediction_file",
                        help="File to save predictions")
    parser.add_argument("--input_file",
                        dest="input_file",
                        help="File with text to summarize")
    parser.add_argument("--question_driven",
                        dest="question_driven",
                        help="Whether to add question to beginning of article for question-driven summarization")
    parser.add_argument("--model_path",
                        dest="model_path",
                        help="Path to model checkpoints")
    parser.add_argument("--model_config",
                        dest="model_config",
                        help="Path to model vocab")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        default=32,
                        help="Batch size for inference")
    return parser


def run_inference():
    """
    Main function for running inference on given input text
    """
    bart = BARTModel.from_pretrained(
        args.model_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=args.model_config
    )

    bart.cuda()
    bart.eval()
    bart.half()
    questions = []
    ref_summaries = []
    gen_summaries = []
    articles = []
    QUESTION_END = " [QUESTION?] "
    with open(args.input_file, 'r', encoding="utf-8") as f:
        source = json.load(f)
    batch_cnt = 0

    for q in tqdm(source):
        question = source[q]['question']
        questions.append(question)
        # The data here may be prepared for the pointer generator, and it is currently easier to 
        # clean the sentence tags out here, as opposed to making tagged and nontagged datasets.
        ref_summary = source[q]['summary']
        if "<s>" in ref_summary:
            ref_summary = ref_summary.replace("<s>", "") 
            ref_summary = ref_summary.replace("</s>", "") 
        ref_summaries.append(ref_summary)
        article = source[q]['articles']
        if args.question_driven == "with_question":
            article = question + QUESTION_END + article
        articles.append(article) 
        # Once the article list fills up, run a batch
        if len(articles) == args.batch_size:
            batch_cnt += 1
            print("Running batch {}".format(batch_cnt))
            # Hyperparameters as recommended here: https://github.com/pytorch/fairseq/issues/1364
            with torch.no_grad():
                predictions = bart.sample(articles, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for pred in predictions:
                #print(pred)
                gen_summaries.append(pred)
            articles = []
            print("Done with batch {}".format(batch_cnt))

    if len(articles) != 0: 
        predictions = bart.sample(articles, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for pred in predictions:
            print(pred)
            gen_summaries.append(pred)

    assert len(gen_summaries) == len(ref_summaries)
    prediction_dict = {
        'question': questions,
        'ref_summary': ref_summaries,
        'gen_summary': gen_summaries
        }

    with open(args.prediction_file, "w", encoding="utf-8") as f:
        json.dump(prediction_dict, f, indent=4)


if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    run_inference()
