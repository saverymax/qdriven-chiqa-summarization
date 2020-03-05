"""
PubMedClient module.

If this module is run (and not imported), it cannot use history,
as the citations will not be written correctly. Useing history could potentially work
if it is imported.
"""

import csv
from Bio import Entrez
import lxml.etree as le
import re
import urllib
import sys
import argparse
from pathlib import Path


class CitationDownloader():
    """
    Class to download citations from entrez
    """

    def __init__(self):
        """
        Initiate the downloader
        """
        Entrez.email = 'max.savery@nih.gov'
        Entrez.api_key = 'e4538172efc6b10d3e54b1bb61dd7a365b08'

    def search_entrez(self, query, history):
        """
        connect with eutils
        """
        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                #datetype='pdat',
                #mindate='2019/1/01',
                #maxdate='2017/2/31',
                retmax=10000,
                #retmax=2000000,
                usehistory=history
                )
            search_results = Entrez.read(search_handle)
            assert type(search_results) != None
            assert isinstance(search_results, dict)
            id_list = search_results['IdList']
            assert isinstance(id_list, list)
            assert all(id_list)
            if len(id_list) == 0:
                return None
            else:
                return search_results

        except Entrez.Parser.CorruptedXMLError:
            print(query_date, search_handle)

    def fetch_with_history(self, search_results):
        """
        Fetch the articles for any given journal using efetch and the ids found by esearch
        using webenv and query key
        """
        id_list = search_results["IdList"]
        count = int(search_results["Count"])
        #print(count)
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]
        max_return = 10000
        step = 10000

        for chunk in range(0, count, step):
            print("downloading ids", chunk, chunk+step)
            response = Entrez.efetch(
                db="pubmed",
                retmode='xml',
                retstart=chunk,
                retmax=max_return,
                webenv=webenv,
                query_key=query_key)

            assert type(response) is not None
            # parse turns it into an ElementTree object with write method,
            # but I can't just write this because it adds the parent <PubmedArticleSet>
            record_tree = le.parse(response)
            record_root = record_tree.getroot()
            for citation in record_root:
                yield citation

    def fetch_without_history(self, search_results, max_return=10000):
        """
        Fetch the articles for any given journal using efetch and the ids found by esearch
        without history
        """
        ids = ",".join(search_results['IdList'])
        response = Entrez.efetch(
            db="pubmed",
            retmax=max_return,
            id=ids,
            retmode='xml')

        assert type(response) is not None
        record_tree = le.parse(response)
        record_root = record_tree.getroot()
        for citation in record_root:
            yield citation
