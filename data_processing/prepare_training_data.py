"""
Module for classes to prepare training datasets

Prepare training data for pointer generator (add tags <s> and </s> tags to summaries):
    Prepare bioasq
    python prepare_training_data.py -bt
    Or with the question:
    python prepare_training_data.py -bt --add-q

Prepare training data for bart, with or without question appended to beginning of abstract text
python prepare_training_data.py --bart-bioasq --add-q
python prepare_training_data.py --bart-bioasq

Prepare training data for sentence classification:
python prepare_training_data.py --bioasq-sent
"""

import json
import argparse
import re
import os

from sklearn.utils import shuffle as sk_shuffle
import rouge
import spacy


def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("-t",
                        dest="tag_sentences",
                        action="store_true",
                        help="tag the sentences with <s> and </s>, for use with pointer generator network")
    parser.add_argument("-e",
                        dest="summ_end_tag",
                        action="store_true",
                        help="Add the summ end tag to the end of the summaries. This was observed not to improve performance on the MedInfo evaluation set")
    parser.add_argument("-b",
                        dest="bioasq_pg",
                        action="store_true",
                        help="Make the bioasq training set for pointer generator")
    parser.add_argument("--bioasq-sent",
                        dest="bioasq_sc",
                        action="store_true",
                        help="Make the bioasq training set for sentence classification")
    parser.add_argument("--bart-bioasq",
                        dest="bart_bioasq",
                        action="store_true",
                        help="Prepare the bioasq training set for bart")
    parser.add_argument("--add-q",
                        dest="add_q",
                        action="store_true",
                        help="Concatenate the question to the beginning of the text. Currently only implemented as an option for bart data and the bioasq abs2summ data")
    return parser


class BioASQ():
    """
    Class to create various versions of BioASQ training dataset
    """

    def __init__(self):
        """
        Initiate spacy
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.Q_END = " [QUESTION?] "
        self.SUMM_END = " [END]"
        self.ARTICLE_END = " [ARTICLE_SEP] "

    def format_summary_sentences(self, summary):
        """
        Split summary into sentences and add sentence tags to the strings: <s> and </s>
        """
        tokenized_abs = self.nlp(summary)
        summary = " ".join(["<s> {s} </s>".format(s=s.text.strip()) for s in tokenized_abs.sents])
        return summary

    def _load_bioasq(self):
        """
        Load bioasq collection generated in process_bioasq.py
        """
        with open("data/bioasq_collection.json", "r", encoding="utf8") as f:
            data = json.load(f)
        return data

    def create_abstract2snippet_dataset(self):
        """
        Generate the bioasq abstract to snippet (1 to 1) training dataset. This function creates data uses the same keys, summary and articles for each summary-article pair
        as the medlinplus training data does. This allows compatibility with the answer summarization data loading function in the pointer generator network.

        This is currently the only dataset with the option to include question. Since this dataset works the best, the add_q option was not added to the others.
        """
        bioasq_collection = self._load_bioasq()
        training_data_dict = {}
        snip_id = 0
        for i, q in enumerate(bioasq_collection):
            question = q
            for snippet in bioasq_collection[q]['snippets']:
                training_data_dict[snip_id] = {}
                if args.summ_end_tag:
                    snippet_text = snippet['snippet'] + self.SUMM_END
                else:
                    snippet_text = snippet['snippet']
                if args.tag_sentences:
                    snippet_text = self.format_summary_sentences(snippet_text)
                training_data_dict[snip_id]['summary'] = snippet_text
                # Add the question with a special question seperator token to the beginning of the article.
                abstract = snippet['article']
                with_question = "_without_question"
                if args.add_q:
                    abstract = question + self.Q_END + abstract
                    with_question = "_with_question"
                training_data_dict[snip_id]['articles'] = abstract
                training_data_dict[snip_id]['question'] = question
                snip_id += 1

        with open("data/bioasq_abs2summ_training_data{}.json".format(with_question), "w", encoding="utf=8") as f:
            json.dump(training_data_dict, f, indent=4)

    def calculate_sentence_level_rouge(self, snip_sen, abs_sen, evaluator):
        """
        For each pair of sentences, calculate rouge score
        """
        rouge_score = evaluator.get_scores(abs_sen, snip_sen)['rouge-l']['f']
        return rouge_score

    def create_binary_sentence_classification_dataset_with_rouge(self):
        """
        Create a dataset for training a sentence classification model, where the binary y labels are assigned based on the
        best rouge score for a sentence in the article when compared to each sentence in the summary
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

        bioasq_collection = self._load_bioasq()
        training_data_dict = {}#
        snip_id = 0
        for i, q in enumerate(bioasq_collection):
            question = q
            for snippet in bioasq_collection[q]['snippets']:
                training_data_dict[snip_id] = {}
                labels = []
                # Sentencize snippet
                snippet_text = snippet['snippet']
                tokenized_snip = self.nlp(snippet_text)
                snippet_sentences = [s.text.strip() for s in tokenized_snip.sents]
                # Sentencize abstract
                abstract_text = snippet['article']
                tokenized_abs = self.nlp(abstract_text)
                abstract_sentences = [s.text.strip() for s in tokenized_abs.sents]
                rouge_scores = []
                for abs_sen in abstract_sentences:
                    best_rouge = 0
                    for snip_sen in snippet_sentences:
                        rouge_score = self.calculate_sentence_level_rouge(snip_sen, abs_sen, evaluator)
                        if best_rouge < rouge_score:
                            best_rouge = rouge_score
                    if best_rouge > .9:
                        label = 1
                    else:
                        label = 0
                    labels.append(label)
                training_data_dict[snip_id]['question'] = q
                training_data_dict[snip_id]['sentences'] = abstract_sentences
                training_data_dict[snip_id]['labels'] = labels
                snip_id += 1

        with open("data/bioasq_abs2summ_binary_sent_classification_training.json", "w", encoding="utf=8") as f:
            json.dump(training_data_dict, f, indent=4)
        # For each sentence in each abstract, compare it to each sentence in answer. Record the best rouge score.

    def create_data_for_bart(self):
        """
        Write the train and val data to file so that the processor and tokenizer for bart will read it, as per fairseqs design
        """
        bioasq_collection = self._load_bioasq()
        # Additional string is added to the question of the beginning of the abstract text
        if args.add_q:
            q_name = "with_question"
        else:
            q_name = "without_question"

        # Open medinfo data preprocessed in prepare_validation_data.py
        with open("data/medinfo_section2answer_validation_data_{}.json".format(q_name), "r", encoding="utf-8") as f:
            medinfo_val = json.load(f)

        try:
            os.mkdir("data/bart/{}".format(q_name))
        except FileExistsError:
            print("Directory ", q_name , " already exists")

        train_src = open("data/bart/{q}/bart.train_{q}.source".format(q=q_name), "w", encoding="utf8")
        train_tgt = open("data/bart/{q}/bart.train_{q}.target".format(q=q_name), "w", encoding="utf8")
        val_src = open("data/bart/{q}/bart.val_{q}.source".format(q=q_name), "w", encoding="utf8")
        val_tgt = open("data/bart/{q}/bart.val_{q}.target".format(q=q_name), "w", encoding="utf8")
        snippets_list = []
        abstracts_list = []
        for i, q in enumerate(bioasq_collection):
            for snippet in bioasq_collection[q]['snippets']:
                snippet_text = snippet['snippet'].strip()
                abstract_text = snippet['article'].strip()
                # Why is there whitespace in the question?
                question = q.replace("\n", " ")
                if args.add_q:
                    abstract_text = question + self.Q_END + abstract_text
                abstracts_list.append(abstract_text)
                snippets_list.append(snippet_text)

        snp_cnt = 0
        print("Shuffling data")
        snippets_list, abstracts_list = sk_shuffle(snippets_list, abstracts_list, random_state=13)
        for snippet_text, abstract_text in zip(snippets_list, abstracts_list):
            snp_cnt += 1
            train_src.write("{}\n".format(abstract_text))
            train_tgt.write("{}\n".format(snippet_text))

        for q_id in medinfo_val:
            # The prepared medinfo data may have sentence tags in it for pointer generator.
            # There is an option in the prepare_validation_data.py script to not tag the data,
            # but it is easier to keep track of the datasets to just remove the tags here.
            summ = medinfo_val[q_id]['summary'].strip()
            summ = summ.replace("<s>", "")
            summ = summ.replace("</s>", "")
            articles = medinfo_val[q_id]['articles'].strip()
            val_src.write("{}\n".format(articles))
            val_tgt.write("{}\n".format(summ))

        train_src.close()
        train_tgt.close()
        val_src.close()
        val_tgt.close()

        # Make sure there were no funny newlines added
        train_src = open("data/bart/{q}/bart.train_{q}.source".format(q=q_name), "r", encoding="utf8").readlines()
        train_tgt = open("data/bart/{q}/bart.train_{q}.target".format(q=q_name), "r", encoding="utf8").readlines()
        val_src = open("data/bart/{q}/bart.val_{q}.source".format(q=q_name), "r", encoding="utf8").readlines()
        val_tgt = open("data/bart/{q}/bart.val_{q}.target".format(q=q_name), "r", encoding="utf8").readlines()
        print("Number of snippets: ", snp_cnt)
        assert len(train_src) == snp_cnt, len(train_src)
        assert len(train_tgt) == snp_cnt
        assert len(val_src) == len(medinfo_val)
        assert len(val_tgt) == len(medinfo_val)


def process_data():
    """
    Save training data sets
    """
    if args.bioasq_pg:
        BioASQ().create_abstract2snippet_dataset()
    if args.bioasq_sc:
        BioASQ().create_binary_sentence_classification_dataset_with_rouge()
    if args.bart_bioasq:
        BioASQ().create_data_for_bart()

if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    process_data()
