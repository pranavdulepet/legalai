import os
from elasticsearch import Elasticsearch
import time

from utils.dataset import DataSetFactory
from search import DocSearch


def get_test_set():
    dataset = DataSetFactory.get_data(name='scdb')
    _, raw_test_corpus, _, _ = dataset.get_data(clean=False)
    _, test_meta_data = dataset.get_metadata()

    limit = -1
    docs = []
    for text, record in zip(raw_test_corpus[:limit], test_meta_data[:limit].iterrows()):
        index, meta = record
        if meta["issue_area"] == None or text == "":
            continue
        entry = {
            "issue": meta["issue"],
            "issue_area": meta["issue_area"],
            "decision_direction": meta["decision_direction"],
            "us_cite_id": meta["us_cite_id"],
            "maj_opinion_author": meta["maj_opinion_author"],
            "argument_date": meta["argument_date"],
            "decision_date": meta["decision_date"],
            "n_min_votes": meta["n_min_votes"],
            "n_maj_votes": meta["n_maj_votes"],
            "case_name": meta["case_name"],
            "text": text,
            }
        docs.append({"_id": index, "_source": entry})
    return docs

if __name__ == "__main__":
    print(os.getenv('ELASTIC_HOST'))
    print(os.getenv('ELASTIC_PORT'))
    time.sleep(60)
    print(f"elastic ping: {Elasticsearch([{'host': os.getenv('ELASTIC_HOST'), 'port': os.getenv('ELASTIC_PORT')}]).ping()}")

    search = DocSearch(host_ip=os.getenv('ELASTIC_HOST'), port=os.getenv('ELASTIC_PORT'), index_name='scdb')
    search.setup()
    docs = get_test_set()
    ret = search.add_docs_bulk(docs)
    print(ret)

    time.sleep(1)
    response = search.search_text(query_text='military')
    print(response.hits.total.value)
    for hit in response:
        for frag in hit.meta.highlight.text:
            print(frag)
        print('------\n')
