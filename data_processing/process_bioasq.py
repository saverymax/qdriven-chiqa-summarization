"""
Script for processing BioASQ json data and saving

to download the pubmed articles for each snippet run
python process_bioasq.py -d
then to process the questions, answers, and snippets, run:
python process_bioasq.py -p
"""


import json
import sys
import os
import argparse
import lxml.etree as le
import glob
from collections import Counter

import numpy as np


def get_args():
    """
    Get command line arguments
    """

    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("-p",
                        dest="process",
                        action="store_true",
                        help="Process bioasq data")
    return parser


class BioASQ():
    """
    Class for processing and saving BioASQ data
    """

    def _load_bioasq(self):
        """
        Load bioasq dataset
        """
        with open("data/BioASQ-training7b/BioASQ-training7b/training7b.json", "r", encoding="ascii") as f:
            bioasq_questions = json.load(f)['questions']
        return bioasq_questions

    def bioasq(self):
        """
        Process BioASQ training data. Generate summary stats. Save questions, ideal answers, snippets, articles, and question types.
        """
        bioasq_questions = self._load_bioasq()
        with open("data/bioasq_pubmed_articles.json", "r", encoding="ascii") as f:
            articles = json.load(f)
        # Dictionary to save condensed json of bioasq
        bioasq_collection = {}
        questions = []
        ideal_answers = []
        ideal_answer_dict = {}
        exact_answers = []
        snippet_dict = {}
        for i, q in enumerate(bioasq_questions):
            # Get the question
            bioasq_collection[q['body']] = {}
            questions.append(q['body'])
            # Get the references used to answer that question
            pmid_list= [d.split("/")[-1] for d in q['documents']]
            # Get the question type: list, summary, yes/no, or factoid
            q_type = q['type']
            bioasq_collection[q['body']]['q_type'] = q_type
            # Take the first ideal answer
            assert isinstance(q['ideal_answer'], list)
            assert isinstance(q['ideal_answer'][0], str)
            ideal_answer_dict[i] = q['ideal_answer'][0]
            bioasq_collection[q['body']]['ideal_answer'] = q['ideal_answer'][0]
            # And get the first exact answer
            if q_type != "summary":
                # Yesno questions will have just a yes/no string in exact answer.
                if q_type == "yesno":
                    exact_answers.append(q['exact_answer'][0])
                    bioasq_collection[q['body']]['exact_answer'] = q['exact_answer'][0]
                else:
                    if isinstance(q['exact_answer'], str):
                        exact_answers.append(q['exact_answer'])
                        bioasq_collection[q['body']]['exact_answer'] = q['exact_answer']
                    else:
                        exact_answers.append(q['exact_answer'][0])
                        bioasq_collection[q['body']]['exact_answer'] = q['exact_answer'][0]
            # Then handle the snippets (the text extracted from the abstract)
            bioasq_collection[q['body']]['snippets'] = []
            snippet_dict[q['body']] = []
            for snippet in q['snippets']:
                pmid_match = False
                snippet_dict[q['body']].append(snippet['text'])
                doc_pmid = str(snippet['document'].split("/")[-1])
                try:
                    article = articles[doc_pmid]
                    # Add the data to the dictionary containing the collection.
                    bioasq_collection[q['body']]['snippets'].append({'snippet': snippet['text'], 'article': article, 'pmid': doc_pmid})
                except KeyError as e:
                    continue

        with open("data/bioasq_ideal_answers.json", "w", encoding="utf8") as f:
            json.dump(ideal_answer_dict, f, indent=4)
        with open("data/bioasq_snippets.json", "w", encoding="utf8") as f:
            json.dump(snippet_dict, f, indent=4)
        with open("data/bioasq_collection.json", "w", encoding="utf8") as f:
            json.dump(bioasq_collection, f, indent=4)

    
def process_bioasq():
    """
    Main processing function for bioasq data
    """
    bq = BioASQ()
    if args.process:
        bq.bioasq()

if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    process_bioasq()
