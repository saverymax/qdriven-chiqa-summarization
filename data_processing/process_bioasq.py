"""
Script for processing BioASQ json data and saving

to download the pubmed articles for each snippet run
python process_bioasq.py -d
then to process the questions, answers, and snippets, run:
python process_bioasq.py -p
or to process all of this but join the snippets that are taken from the same
abstract but are listed separately in the file:
python process_bioasq.py -pj
"""


import json
import sys
import os
import argparse
import lxml.etree as le
import glob
from collections import Counter

import numpy as np
import spacy

import PubMedClient


def get_args():
    """
    Get command line arguments
    """

    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("-p",
                        dest="process",
                        action="store_true",
                        help="Process bioasq data")
    parser.add_argument("-j",
                        dest="join_snippets",
                        action="store_true",
                        help="Join the snippets from the same abstract")
    parser.add_argument("-d",
                        dest="download",
                        action="store_true",
                        help="Download bioasq pubmed articles")

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
            pmid_dict = {}
            unique_abs_index = 0
            for snippet in q['snippets']:
                pmid_match = False
                snippet_dict[q['body']].append(snippet['text'])
                if q['body'] == "List signaling molecules (ligands) that interact with the receptor EGFR?":
                    print(snippet_dict[q['body']])
                doc_pmid = str(snippet['document'].split("/")[-1])
                if args.join_snippets:
                    # If the abstract has already been processed along with its snippet
                    # find the location of that snippet so I can add a new snippet
                    if doc_pmid in pmid_dict:
                        pmid_match = True
                        snippet_index = pmid_dict[doc_pmid]
                    else:
                        # Check to make sure the abstract was collected, because
                        # if it wasn't, the code below will not add that abstract to the list,
                        # unsurprisingly.
                        if doc_pmid in articles:
                            pmid_dict[doc_pmid] = unique_abs_index
                            unique_abs_index += 1
                try:
                    if args.join_snippets and pmid_match:
                        snippet_text = bioasq_collection[q['body']]['snippets'][snippet_index]['snippet']
                        snippet_text += " " + snippet['text']
                        bioasq_collection[q['body']]['snippets'][snippet_index]['snippet'] = snippet_text
                    else:
                        article = articles[doc_pmid]
                        # Add the data to the dictionary containing the collection.
                        bioasq_collection[q['body']]['snippets'].append({'snippet': snippet['text'], 'article': article, 'pmid': doc_pmid})
                except KeyError as e:
                    print("No article found for this snippet", e)
                    continue

        with open("data/bioasq_ideal_answers.json", "w", encoding="utf8") as f:
            json.dump(ideal_answer_dict, f, indent=4)
        with open("data/bioasq_snippets.json", "w", encoding="utf8") as f:
            json.dump(snippet_dict, f, indent=4)
        if args.join_snippets:
            with open("data/bioasq_collection_with_joined_snippets.json", "w", encoding="utf8") as f:
                json.dump(bioasq_collection, f, indent=4)
        else:
            with open("data/bioasq_collection.json", "w", encoding="utf8") as f:
                json.dump(bioasq_collection, f, indent=4)

    def get_bioasq_docs(self):
        """
        Download and save bioasq articles, for use while processing other parts of bioasq data.
        """
        bioasq_questions = self._load_bioasq()
        documents = {}
        for i, q in enumerate(bioasq_questions):
            pmid_list= [d.split("/")[-1] for d in q['documents']]
            result = self._download_bioasq_docs(pmid_list)
            # 27924029 is the only document for one question, and
            # is no longer in pubmed.
            if result is None:
                pass
            else:
                if len(result) != len(pmid_list):
                    for i in pmid_list:
                        if i not in result:
                            print("No article returned fron PubMed, or missing title/abstract: ", i)
                documents.update(result)

        with open("data/bioasq_pubmed_articles.json", "w", encoding="utf8") as f:
            json.dump(documents, f, indent=4)

    def _download_bioasq_docs(self, pmid_list):
        """
        If command line argument included, download the documents specified by BioASQ
        """
        history = "n"
        query = "[UID] OR ".join(pmid_list)
        query += "[UID]"
        id_cnt=0
        doc_dict = {}

        downloader = PubMedClient.CitationDownloader()
        search_results = downloader.search_entrez(query, history)

        if search_results is None:
           print("No ids")
           print(pmid_list)
           return None

        if history == "y":
            fetched_results = downloader.fetch_with_history(search_results)
        elif history == "n":
            fetched_results = downloader.fetch_without_history(search_results)

        for citation in fetched_results:
            try:
                pmid = citation.find(".//PMID").text
                id_cnt+=1
                title = citation.find(".//ArticleTitle")
                if title is not None:
                    title = le.tostring(title, encoding='unicode', method='text').strip().replace("\n", " ")
                else:
                    title = ""
                    print("No title", pmid)
                    continue
                abstract = citation.find(".//Abstract")
                if abstract is not None:
                    abstract = le.tostring(abstract, encoding='unicode', method='text').strip().replace("\n", " ")
                else:
                    abstract = ""
                    print("No abstract", pmid)
                    continue
                text =  title + " " + abstract
                doc_dict[pmid] = text

            except Exception as e:
                raise e

        return doc_dict


def process_bioasq():
    """
    Main processing function for bioasq data
    """
    bq = BioASQ()
    if args.download:
        bq.get_bioasq_docs()
    if args.process:
        bq.bioasq()

if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    process_bioasq()
