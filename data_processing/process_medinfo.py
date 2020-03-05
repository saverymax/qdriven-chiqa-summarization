"""
This module will process 1. the medinfo data, and 2. search the chiqa collection for the corresponding pages

If the -m option is provided the script will:

    parse medinfo tsv and save it in json, includng questions, answers, urls and section titles

    Generate the medinfo_data.json file, the medinfo data converted and cleaned from tsv to json.
    This file will be used to create the medinfo_collection.json file.

    All sections and urls without values will be labeled as "N/A"

If the -c option is included:

    The module will parse chiqa crawl collection and find answer pages/sections for corresponding summaries/answers in
    the MedInfo collection, as well as the collection for CHIQA evaluation

    This requires process_medinfo.py and process_chiqa_answers.py to have already been run, generating json files
    of the data.

Example pipeline:
python process_medinfo.py -m
Then copy the relevant chiqa and dailymed pages to a new directory
python process_medinfo.py -cy
and parse the data from those files and save as text:
python process_medinfo.py -cp
or parse and save as xml
python process_medinfo.py -cpx
"""

import json
import sys
import os
import argparse
import lxml.etree as le
import re
import glob
import shutil
from collections import Counter

import pandas as pd
import numpy as np
import spacy


def get_args():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("-m",
                        dest="medinfo",
                        action="store_true",
                        help="Process the medinfo spreadsheet and save the data to json")
    parser.add_argument("-c",
                        dest="chiqa",
                        action="store_true",
                        help="Process the chiqa Data. Use with options -y or -p")
    parser.add_argument("-y",
                        dest="copy_files",
                        action="store_true",
                        help="Copy files from the chiqa collection relevant for medinfo")
    parser.add_argument("-p",
                        dest="parse_files",
                        action="store_true",
                        help="Parse the sections from the relevant chiqa files")
    parser.add_argument("-x",
                        dest="save_xml",
                        action="store_true",
                        help="Save the xml as xml instead of text in the medinfo collection")
    return parser


class ProcessMedInfo():
    """
    Medinfo processor
    """

    def write_medinfo_stats(self, q_cnt, avg_q_sent, avg_q_tok, a_cnt, avg_a_sent, avg_a_tok, q_match, unique_urls, unique_sites, site_cnt):
        """
        Write medinfo stats
        """
        with open("results/summary_file_medinfo.txt", "w", encoding="utf-8") as summary_file:
            summary_file.write(
                "\nMedInfo\nNumber of Questions: {qc}\nAverage question tokens: {qt}\nAverage question sentences: {qs}\nNumber of answers: {ac}\nAverage answer tokens: {at}\nAverage answer sentences: {ase}\nQuestion duplicates: {ud}\nUnique URLS: {uu}\n".format(
                    qc=q_cnt, qt=avg_q_tok, qs=avg_q_sent, ac=a_cnt, at=avg_a_tok, ase=avg_a_sent, ud=q_match, uu=unique_urls
                    )
                )
            summary_file.write(
                "\nNumber of unique sites: {sc}\nCount of sites: {us}\n".format(
                    sc=site_cnt, us=unique_sites
                    )
                )

    def process_medinfo(self):
        """
        Load and process medinfo excel data
        """
        df = pd.read_excel("../data/MedInfo2019-QA-Medications.xlsx", encoding="latin", na_filter=False) # na_filter=false to deal with none-like stuff on my own
        print("Shape of MedInfo data:", df.shape)
        self.nlp = spacy.load('en_core_web_sm')
        data_dict = {}
        total_q_tokens = 0
        total_q_sents = 0
        total_a_tokens = 0
        total_a_sents = 0
        q_cnt = 0
        a_cnt = 0
        q_match = 0
        url_dict = {'urls': []}
        sites = []

        for i, row in df.iterrows():
            try:
                q_cnt += 1
                a_cnt += 1
                question = row['Question']
                answer_url = row['URL']
                if answer_url == "n/a" or answer_url == "NA" or answer_url is np.nan:
                    sites.append("N/A")
                else:
                    site = answer_url.split("//")[1].split(".")
                    if site[0] == "www":
                        sites.append(site[1])
                    else:
                        sites.append(site[0])
                # Note: Some answers will be no answer or unanswerable
                answer = row['Answer']
                # There are question with multiple answers
                if question in data_dict:
                    q_match += 1
                if question not in data_dict:
                    data_dict[question] = []
                # Handle answers with None type strings
                if answer_url == "n/a" or answer_url == "NA" or answer_url is np.nan:
                    answer_url = "N/A"
                # List to track unique urls and to make json of unique urls
                url_dict['urls'].append(answer_url)
                # Handle sections with None type strings
                section = row['Section Title'] # Some sections will have n/a or none type strings. This will be dealt with later.
                if section == "n/a" or section == "NA" or section is np.nan or section == "none" or section == "":
                    section = "N/A"

                tokenized_a = self.nlp(answer)
                total_a_sents += len([s for s in tokenized_a.sents])
                total_a_tokens += len([t for t in tokenized_a])
                tokenized_q = self.nlp(question)
                total_q_sents += len([s for s in tokenized_q.sents])
                total_q_tokens += len([t for t in tokenized_q])

                data_dict[question].append({'url': answer_url, 'answer': answer, 'focus': row['Focus (Drug)'], 'type': row['Question Type'], 'section_title': section})

            except Exception as e:
                print("Error!: ", answer_url, q_cnt)
                raise

        print("Saving medinfo data as json...")
        with open("data/medinfo_data.json", "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)
        url_dict['urls'] = list(set(url_dict['urls']))
        with open("data/medinfo_answer_urls.json", "w", encoding="utf-8") as f:
            json.dump(url_dict, f, indent=4)

        avg_q_sent = total_q_sents / q_cnt
        avg_q_tok = total_q_tokens / q_cnt
        avg_a_sent = total_a_sents / a_cnt
        avg_a_tok = total_a_tokens / a_cnt
        unique_sites = Counter(sites)
        self.write_medinfo_stats(q_cnt, avg_q_sent, avg_q_tok, a_cnt, avg_a_sent, avg_a_tok, q_match, len(set(url_dict['urls'])), unique_sites, len(set(sites)))


class ProcessCrawl():
    """
    Class to parse crawl xml
    """

    def __init__(self):
        """
        Initiate counting variables, data, and spacy
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.section_cnt = 0
        self.page_cnt = 0
        self.n_sentences = 0
        self.n_tokens = 0
        self.page_tokens = 0
        self.page_sentences = 0
        with open("data/medinfo_answer_urls.json", "r", encoding="utf-8") as u:
            self.answer_urls = json.load(u)['urls']
        with open("data/medinfo_data.json", "r", encoding="utf-8") as m:
            self.medinfo_dict = json.load(m)

    def write_medinfo_stats(self):
        """
        Write  medinfo stats
        """
        with open("results/summary_file_medinfo.txt", "a", encoding="utf-8") as summary_file:
            summary_file.write(
                "\nNumber of pages that matched an answer: {pm}\nNumber of sections: {sc}\nSentences per section: {ss}\nTokens per section {ts}\n".format(
                    pm=self.page_cnt, sc=self.section_cnt, ss=self.n_sentences/self.section_cnt, ts=self.n_tokens/self.section_cnt
                    )
                )
            summary_file.write(
                "\nAverage sentences on each pages: {ps}\n Average Tokens on each page: {pt}".format(
                    pt=self.page_tokens / self.page_cnt, ps=self.page_sentences / self.page_cnt
                    )
                )

    def _copy_answer_pages(self, page):
        """
        Search through the chiqa collection. Copy pages that contain answer to a new directory

        The pages are selected based on url matches. Once the subset of the ChiQA collection is created,
        this will be parsed and the answer sections extracted if a different method.
        """
        try:
            tree = le.parse(page)
            doc = tree.getroot()
            url = doc.attrib['url']
            if url in self.answer_urls:
                shutil.copy(page, "Q:\\answer_summ\\data\\medinfo_answer_summ_collection")

        except le.XMLSyntaxError as e:
            print(e, page)

        except Exception as e:
            print("\nERROR:\n", doc.attrib['corpus'], doc.attrib['id'], url, "\n")
            raise

    def _process_xml_text_collection(self, page):
        """
        Parse the xml from each page and save as text

        ENCODING ISSUE? \\u?
        """
        try:
            tree = le.parse(page)
            doc = tree.getroot()
            url = doc.attrib['url']
            corpus = doc.attrib['corpus']
            if url == "N/A":
                print("No url!")
                return
            else:
                # Iterate through each question in medinfo data:
                for question in self.medinfo_dict:
                    # Then iterate through each answer, and check each URL to see if it matches the current document.
                    # If so, get the appropriate section from the xml.
                    for i, answer in enumerate(self.medinfo_dict[question]):
                        if url == answer['url']:
                            self.page_cnt += 1
                            full_page = le.tostring(doc, encoding='utf-8', method='text')
                            full_page = re.sub(r"\s+", " ", full_page.decode("utf-8"))
                            tokenized_doc = self.nlp(full_page)
                            self.page_sentences += len([s for s in tokenized_doc.sents])
                            self.page_tokens += len([t for t in tokenized_doc])
                            self.medinfo_dict[question][i]['full_text'] = full_page
                            section = answer['section_title']
                            if section == "N/A":
                                print("No section! Only full text will be used")
                            else:
                                # This xpath finds the title node with the matching section text. It then goes up to the parent <section> and grabs both the <title> and the <text>
                                matched_section = doc.xpath(".//title[text()='{}']/../*".format(section))
                                # Here, check for string matches between recorded section (human) and text in the document. The human recorded
                                # section was not always recorded perfectly and thus there are a fair amount of mismatches that I do not handle.
                                if matched_section == []:
                                    # Do a partial string match for the section string in question:
                                    found_section = False
                                    title_sections = doc.findall(".//title")
                                    if title_sections != []:
                                        for title in title_sections:
                                            if title.text is not None:
                                                if section.lower() in title.text.lower():
                                                    title_parent = title.getparent()
                                                    section_text = title_parent.find(".//text")
                                                    section_text = le.tostring(section_text, encoding='unicode', method='text')
                                                    print("After search for partial match of section string, using section from \n", page, "\n", url, "\n")
                                                    found_section = True
                                                    break
                                    if found_section == False:
                                        print("No match", doc.attrib['corpus'], page, doc.attrib['url'], section)
                                        continue
                                elif len(matched_section) != 2 and corpus != "DailyMed":
                                    print("Section too long", doc.attrib['corpus'], doc.attrib['id'], doc.attrib['url'], section)
                                    continue
                                else:
                                    section_text = le.tostring(matched_section[1], encoding='unicode', method='text')
                                if len(section_text) < len(answer['answer']):
                                    print("Found section is shorter than the answer")
                                    continue
                                self.section_cnt += 1
                                section_text = re.sub(r"\s+", " ", section_text)
                                self.medinfo_dict[question][i]['section_text'] = section_text
                                # Tokenize and count n tokens and sentences:
                                tokenized_doc = self.nlp(section_text)
                                self.n_sentences += len([s for s in tokenized_doc.sents])
                                self.n_tokens += len([t for t in tokenized_doc])

                                assert len(full_page) > len(answer['answer'])
                                assert len(full_page) > len(section_text)
                                #for chunk in doc.noun_chunks:
                                #    print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)

        except le.Error as e:
            print(e, page, section)

        except Exception as e:
            print("\nERROR:\n", doc.attrib['corpus'], url, page, "\n")
            raise

    def _process_xml_collection(self, page):
        """
        Parse the xml from each page and save as xml

        ENCODING ISSUE? \\u?
        """
        try:
            tree = le.parse(page)
            doc = tree.getroot()
            url = doc.attrib['url']
            corpus = doc.attrib['corpus']
            if url == "N/A":
                print("No url!")
                return
            else:
                # Iterate through each question in medinfo data:
                for question in self.medinfo_dict:
                    # Then iterate through each answer, and check each URL to see if it matches the current document.
                    # If so, get the appropriate section from the xml.
                    for i, answer in enumerate(self.medinfo_dict[question]):
                        if url == answer['url']:
                            self.page_cnt += 1
                            full_xml_page = le.tostring(doc, encoding='unicode', method='xml')
                            self.medinfo_dict[question][i]['full_text'] = full_xml_page
                            section = answer['section_title']
                            if section == "N/A":
                                print("No section! Only full text will be used")
                            else:
                                # This xpath finds the title node with the matching section text. It then goes up to the parent <section> and grabs both the <title> and the <text>
                                matched_section = doc.xpath(".//title[text()='{}']/../*".format(section))
                                # Here, check for string matches between recorded section (human) and text in the document. The human recorded
                                # section was not always recorded perfectly and thus there are a fair amount of mismatches that I do not handle.
                                if matched_section == []:
                                    # Do a partial string match for the section string in question:
                                    found_section = False
                                    title_sections = doc.findall(".//title")
                                    if title_sections != []:
                                        for title in title_sections:
                                            if title.text is not None:
                                                if section.lower() in title.text.lower():
                                                    title_parent = title.getparent()
                                                    section_text = title_parent.find(".//text")
                                                    section_xml = le.tostring(section_text, encoding='unicode', method='xml')
                                                    section_text = le.tostring(section_text, encoding='unicode', method='text')
                                                    print("After search for partial match of section string, using section from \n", page, "\n", url, "\n")
                                                    found_section = True
                                                    break
                                    if found_section == False:
                                        print("No match", doc.attrib['corpus'], page, doc.attrib['url'], section)
                                        continue
                                elif len(matched_section) != 2 and corpus != "DailyMed":
                                    print("Section too long", doc.attrib['corpus'], doc.attrib['id'], doc.attrib['url'], section)
                                    continue
                                else:
                                    section_xml = le.tostring(matched_section[1], encoding='unicode', method='xml')
                                    section_text = le.tostring(matched_section[1], encoding='unicode', method='text')
                                if len(section_text) < len(answer['answer']):
                                    print("Found section is shorter than the answer")
                                    continue
                                self.section_cnt += 1
                                section_text = re.sub(r"\s+", " ", section_text)
                                self.medinfo_dict[question][i]['section_text'] = section_xml
                                # Tokenize and count n tokens and sentences:

        except le.Error as e:
            print(e, page, section)

        except Exception as e:
            print("\nERROR:\n", doc.attrib['corpus'], url, page, "\n")
            raise

    def copy_chiqa_files(self):
        """
        Iterate through the directories and the files in each

        There is a directory for each source of information
        """
        for f in glob.iglob("../data/crawl_download_asumm/*"):
            print("Directory: {}".format(f))
            for xml_file in glob.iglob("{f}/*.xml".format(f=f)):
                self._copy_answer_pages(xml_file)
        # Dailymed parsing:
        print("DailyMed crawl:\n")
        for xml_file in glob.iglob("../data/xml_dailymed_12072018/xml_dailymed/*.xml"):
        #for xml_file in glob.iglob("../data/xml_dailymed_12072018/xml_dailymed/20160817_8ae4a0c1-1424-47a9-9a59-7fe38bedc0c7.xml"):
            self._copy_answer_pages(xml_file)

    def parse_chiqa_files(self):
        """
        Iterate through the xml in the answer summ collection created from the files in crawl_download and in the dailymed crawl
        """
        for xml_file in glob.iglob("../data/medinfo_answer_summ_collection/*.xml"):
            if args.save_xml:
                self._process_xml_collection(xml_file)
            else:
                self._process_xml_text_collection(xml_file)

    def write_data(self):
        """
        Write medinfo dict now with documents
        """
        if args.save_xml:
            with open("data/medinfo_xml_collection.json", "w", encoding="utf8") as f:
                json.dump(self.medinfo_dict, f, indent=4)
        else:
            with open("data/medinfo_collection.json", "w", encoding="utf8") as f:
                json.dump(self.medinfo_dict, f, indent=4)

def run_processing():
    """
    Main function to run crawl processing
    """
    if args.medinfo:
        processor = ProcessMedInfo()
        processor.process_medinfo()
    elif args.chiqa:
        crawl = ProcessCrawl()
        if args.copy_files:
            crawl.copy_chiqa_files()
        elif args.parse_files:
            crawl.parse_chiqa_files()
            crawl.write_data()
            if not args.save_xml:
                crawl.write_medinfo_stats()
        else:
            sys.stderr.write("No arguments supplied. Module will now exit")
            sys.exit()


if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    run_processing()
