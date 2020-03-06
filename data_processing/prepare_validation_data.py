"""
Module for classes to prepare validation dataset from MedInfo dataset.
Data format will be {key: {'question': question, 'summary':, summ, 'articles': articles} ...}

Additionally, format for question driven summarization. For example:
python prepare_validation_data.py -t --add-q
"""


import json
import argparse
import re

import spacy
import rouge

def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--pg",
                        dest="pg",
                        action="store_true",
                        help="tag the sentences with <s> and </s>, for use with pointer generator network")
    parser.add_argument("--bart",
                        dest="bart",
                        action="store_true",
                        help="Prepare data for BART")
    parser.add_argument("--add-q",
                        dest="add_q",
                        action="store_true",
                        help="Concatenate the question to the beginning of the text for question driven summarization")

    return parser


class MedInfo():

    def __init__(self):
        """
        Initiate class for processing medinfo collection
        """
        self.nlp = spacy.load('en_core_web_sm')
        if args.add_q:
            self.q_name = "_with_question"
        else:
            self.q_name = "_without_question"

    def _load_collection(self):
        """
        Load medinfo collection prepared in the process_medinfo.py script
        """
        with open("data/medinfo_collection.json", "r", encoding="utf-8") as f:
            medinfo = json.load(f)

        return medinfo

    def _format_summary_sentences(self, summary):
        """
        Split summary into sentences and add sentence tags to the strings: <s> and </s>
        """
        tokenized_abs = self.nlp(summary)
        summary = " ".join(["<s> {s} </s>".format(s=s.text.strip()) for s in tokenized_abs.sents])
        return summary

    def save_section2answer_validation_data(self, tag_sentences):
        """
        For questions that have a corresponding section-answer pair, save the
        validation data in following format 
        {'question': {'summary': text, 'articles': text}}
        """
        dev_dict = {}
        medinfo = self._load_collection()
        data_pair = 0
        Q_END = " [QUESTION?] "
        for i, question in enumerate(medinfo):
            try:
                # There may be multiple answers per question, but for the sake of the validation set,
                # just use the first answer
                if 'section_text' in medinfo[question][0]:
                    article = medinfo[question][0]['section_text']
                    summary = medinfo[question][0]['answer']
                    # Stripping of whitespace was done in processing script for section and full page
                    # but not for answer or question
                    summary = re.sub(r"\s+", " ", summary)
                    question = re.sub(r"\s+", " ", question)
                    if args.add_q:
                        article = question + Q_END + article
                    assert len(summary) <= (len(article) + 10)
                    if tag_sentences:
                        summary = self._format_summary_sentences(summary)
                        tag_string = "_s-tags"
                    else:
                        tag_string = ""
                    data_pair += 1
                    dev_dict[i] = {'question': question, 'summary': summary, 'articles': article}
            except AssertionError:
                print("Answer longer than summary. Skipping element")

        print("Number of page-section pairs:", data_pair)

        with open("data/medinfo_section2answer_validation_data{0}{1}.json".format(self.q_name, tag_string), "w", encoding="utf-8") as f:
            json.dump(dev_dict, f, indent=4)


def process_data():
    """
    Main function for saving data
    """
    # Run once for each 
    if args.pg:
        MedInfo().save_section2answer_validation_data(tag_sentences=True)
    if args.bart:
        MedInfo().save_section2answer_validation_data(tag_sentences=False)

if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    process_data()
