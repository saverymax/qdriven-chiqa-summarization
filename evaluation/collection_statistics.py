"""
Script to compute statistics on collection and validate the data integrity

To run with counts of sentences and tokens:
python collection_statistics.py --tokenize
Otherwise don't include --tokenize option
"""

import argparse
import spacy
import json
import numpy as np
from collections import Counter


def get_args():
    """
    Argument parser for preparing chiqa data
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--tokenize",
                        dest="tokenize",
                        action="store_true",
                        help="Tokenize by words and sentences, counting averages/sd for each.")
    return parser


class SummarizationDataStats():
    """
    Class for validating annotated collection
    """

    def __init__(self):
        """
        Init spacy and counting variables.
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.summary_file = open("results/question_driven_answer_summ_collection_stats.txt", "w", encoding="utf-8")

    def load_data(self, dataset, dataset_name):
        """
        Given the path of the dataset, load it!
        """
        with open(dataset, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.dataset_name = dataset_name

    def _get_token_cnts(self, doc, doc_type):
        """
        Count tokens and in documents
        """
        tokenized_doc = self.nlp(doc)
        self.stat_dict[doc_type][0].append(len([s for s in tokenized_doc.sents]))
        doc_len = len([t for t in tokenized_doc])
        self.stat_dict[doc_type][1].append(doc_len)
        if doc_len < 50 and doc_type == "answer":
            print("Document less than 50 tokens:", url)

    def iterate_data(self):
        """
        Count the number of examples in each dataset
        """
        if "single" in self.dataset_name:
            # Index 0 for list of sentence lengths, index 1 for list of token lengths
            self.stat_dict = {'question': [[], []], 'summary': [[], []], 'article': [[], []]}
            for answer_id in self.data:
                summary = self.data[answer_id]['summary']
                articles = self.data[answer_id]['articles']
                question = self.data[answer_id]['question']
                if args.tokenize:
                    self._get_token_cnts(summary, 'summary')
                    self._get_token_cnts(articles, 'article')
                    self._get_token_cnts(question, 'question')
            self._write_stats("token_counts")

        if "multi" in self.dataset_name:
            self.stat_dict = {'question': [[], []], 'summary': [[], []], 'article': [[], []]}
            for q_id in self.data:
                summary = self.data[q_id]['summary']
                question = self.data[q_id]['question']
                if args.tokenize:
                    self._get_token_cnts(summary, 'summary')
                    self._get_token_cnts(question, 'question')
                question = self.data[q_id]['question']
                for answer_id in self.data[q_id]['articles']:
                    articles = self.data[q_id]['articles'][answer_id][0]
                    if args.tokenize:
                        self._get_token_cnts(articles, 'article')
            self._write_stats("token_counts")

        if self.dataset_name == "complete_dataset":
            self.stat_dict = {'urls': [], 'sites': []}
            article_dict = {}
            print("Counting answers, sites, unique urls, and tokenized counts of unique articles")
            answer_cnt = 0
            for q_id in self.data:
                for a_id in self.data[q_id]['answers']:
                    answer_cnt += 1
                    url = self.data[q_id]['answers'][a_id]['url']
                    article = self.data[q_id]['answers'][a_id]['article']
                    if url not in article_dict:
                        article_dict[url] = article
                    self.stat_dict['urls'].append(url)
                    assert "//" in url, url
                    site = url.split("//")[1].split("/")
                    self.stat_dict['sites'].append(site[0])
            print("# of Answers:", answer_cnt)
            print("Unique articles: ", len(article_dict)) # This should match up with count written to file
            self._write_stats("full collection")

            # Get token/sent averages of unique articles
            if args.tokenize:
                self.stat_dict = {'article': [[], []]}
                for a in article_dict:
                    self._get_token_cnts(article_dict[a], 'article')
                self._write_stats("token_counts")

    def _write_stats(self, stat_type, user=None, summ_type=None):
        """
        Return chiqa page stats
        """
        if stat_type == "full collection":
            self.summary_file.write("\n\nDataset: {c}\n".format(c=self.dataset_name))
            self.summary_file.write("Number of unique urls: {u}\nNumber of unique sites: {s}\n".format(u=len(set(self.stat_dict['urls'])), s=len(set(self.stat_dict['sites'])))
                    )
            site_cnts = Counter(self.stat_dict['sites']).most_common()
            for site in site_cnts:
                self.summary_file.write("{s}: {n}\n".format(s=site[0], n=site[1]))

        if stat_type == "token_counts":
            self.summary_file.write("\n\nDataset: {c}\n".format(c=self.dataset_name))
            for doc_type in self.stat_dict:
                if user is not None:
                    self.summary_file.write("\n{0}, {1}\n".format(user, summ_type))

                self.summary_file.write(
                    "\nNumber of {d}s: {p}\nAverage tokens/{d}: {t}\nAverage sentences/{d}: {s}\n".format(
                        d=doc_type, p=len(self.stat_dict[doc_type][0]), t=sum(self.stat_dict[doc_type][1])/len(self.stat_dict[doc_type][1]), s=sum(self.stat_dict[doc_type][0])/len(self.stat_dict[doc_type][0])
                        )
                    )

                self.summary_file.write(
                    "Median tokens/{d}: {p}\nStandard deviation tokens/{d}: {t}\n".format(
                        d=doc_type, p=np.median(self.stat_dict[doc_type][1]), t=np.std(self.stat_dict[doc_type][1])
                        )
                    )

                self.summary_file.write(
                    "Median sentences/{d}: {p}\nStandard deviation sentences/{d}: {t}\n".format(
                        d=doc_type, p=np.median(self.stat_dict[doc_type][0]), t=np.std(self.stat_dict[doc_type][0])
                        )
                    )


def get_stats():
    """
    Main function for getting CHiQA collection stats
    """
    datasets = [
        ("../data_processing/data/page2answer_single_abstractive_summ.json", "p2a-single-abs"),
        ("../data_processing/data/page2answer_single_extractive_summ.json", "p2a-single-ext"),
        ("../data_processing/data/section2answer_multi_abstractive_summ.json", "s2a-multi-abs"),
        ("../data_processing/data/page2answer_multi_extractive_summ.json", "p2a-multi-ext"),
        ("../data_processing/data/section2answer_single_abstractive_summ.json", "s2a-single-abs"),
        ("../data_processing/data/section2answer_single_extractive_summ.json", "s2a-single-ext"),
        ("../data_processing/data/section2answer_multi_extractive_summ.json", "s2a-multi-ext"),
        ("../data_processing/data/question_driven_answer_summarization_primary_dataset.json", "complete_dataset"),
        ]

    stats = SummarizationDataStats()
    for dataset in datasets:
        print(dataset[1])
        stats.load_data(dataset[0], dataset[1])
        stats.iterate_data()


if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    get_stats()
