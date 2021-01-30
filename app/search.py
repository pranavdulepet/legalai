import os
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from elasticsearch_dsl import Search
from elasticsearch_dsl import Q

from utils.dataset import DataSetFactory

# TODO: clean up
class DocSearch:

    SCDB_MAPPING = {
                "properties": {
                    "issue": {"type": "keyword"},
                    "issue_area": {"type": "integer"},
                    "decision_direction": {"type": "keyword"},
                    "us_cite_id": {"type": "keyword"},
                    "maj_opinion_author": {"type": "integer"},
                    "argument_date": {"type": "date"},
                    "decision_date": {"type": "date"},
                    "n_min_votes": {"type": "integer"},
                    "n_maj_votes": {"type": "integer"},
                    "case_name": {"type": "text"},
                    "text": {"type": "text"},
                    "topics": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "ners": {"type": "keyword"},
                    "ext_summary": {"type": "text"},
                }
            }

    def __init__(self, host_ip='localhost', port=9200, index_name='scdb'):
        self.es = Elasticsearch([{'host': host_ip, 'port': port}])
        self.index_name = 'scdb'
        self.dataset = DataSetFactory.get_data(name=index_name)

    def ping(self):
        if self.es.ping():
            print(f"Elastic Search is Active")
            return True
        else:
            print(f"Elastic Search is Inactive")
            return False

    def delete_index(self):
        if self.es.indices.exists(self.index_name):
            print(f'{self.index_name} index exists, deleting...')
            self.es.indices.delete(self.index_name, ignore=[400, 404])
            print(f"deleted {self.index_name}")

    def create_index(self):
        print(f'Creating index: {self.index_name}')
        self.es.indices.create(index=self.index_name, ignore=400)

        if self.es.indices.exists(self.index_name):
            print(f'{self.index_name} created')
            return True
        else:
            print(f'couldnt create {self.index_name}')
            return False

    def create_mapping(self, mapping=SCDB_MAPPING):
        print(f'creating mapping for {self.index_name}')
        ret = self.es.indices.put_mapping(index=self.index_name, body=mapping)
        if ret['acknowledged']:
            print(f'mapping created')
            return True
        else:
            print(f'mapping failed')
            return False

    def setup(self, delete_if_exist=False):
        if self.es.ping():
            if delete_if_exist:
                self.delete_index()
            elif self.es.indices.exists(self.index_name):
                print(f"Index {self.index_name} already exists, returning")
                return False
            if self.create_index():
                return self.create_mapping()

    def add_docs_bulk(self, docs):
        # TODO: test adding docs at run time.
        ret = helpers.bulk(self.es, docs, index=self.index_name)
        return ret

    def add_doc(self, doc_id=None, entry=None):
        # TODO: test adding docs at run time.
        ret = self.es.create(index=self.index_name, id=doc_id, body=entry)
        return ret

    def search_text(self,
                    query_text='tribal wekfare',
                    highlight=True,
                    search_fields = ["text"],
                    search_fields_any=True,
                    search_type="normal",
                    filter_author=None,
                    filter_decision_date=None,
                    filter_argument_date=None,
                    filter_min_votes=None,
                    filter_maj_votes=None,
                    filter_decision_direction=None,
                    filter_cite_id=None,
                    filter_issue=None,
                    filter_issue_area=None,
                    sort_by=None,
                    agg_by=None):
        # TODO: break this down into sub functions.
        # TODO: error handling.
        # TODO: handle pagination.
        s = Search(using=self.es, index=self.index_name)

        s.source([
                    "issue",
                    "issue_area",
                    "decision_direction",
                    "us_cite_id",
                    "maj_opinion_author",
                    "argument_date",
                    "decision_date",
                    "n_min_votes",
                    "n_maj_votes",
                    "case_name",
                ])

        if query_text != "" and len(search_fields) > 0:
            if search_type == "wildcard": # wildcard match
                print(f'doing wildcard search')
                if search_fields_any:
                    print(f'in any fields')
                    q = Q("wildcard", text=query_text) | Q("wildcard", case_name=query_text)
                    s = s.query(q)
                else:
                    print(f"in both fields")
                    for f in search_fields:
                        if f == "text":
                            s = s.query("wildcard", text=query_text)
                        if f == "case_name":
                            s = s.query("wildcard", case_name=query_text)
            elif search_type == "phrase": # match phrase
                print(f'doing phrase match')
                if search_fields_any:
                    print(f'in any fields')
                    q = Q("match_phrase", text=query_text) | Q("match_phrase", case_name=query_text)
                    s = s.query(q)
                else:
                    print(f"in both fields")
                    for f in search_fields:
                        if f == "case_name":
                            s = s.query("match_phrase", case_name=query_text)
                        if f == "text":
                            s = s.query("match_phrase", text=query_text)

            # if not phrase, then do normal match for any number of fields
            elif search_type == "fuzzy":
                print(f'doing fuzzy search')
                if search_fields_any:
                    print(f'doing fuzzy search multi match')
                    s = s.query("multi_match", query=query_text, fields=search_fields, fuzziness="auto", operator="and")
                    # s = s.query("multi_match", case_name={"query":query_text, "operator":'and', "fuzziness":"auto"})
                else:
                    print(f'doing fuzzy search match all fields')
                    for f in search_fields:
                        if f == "case_name":
                            # s = s.query("match", case_name=query_text)
                            s = s.query("match", case_name={"query":query_text, "operator":'and', "fuzziness":"auto"})
                        if f == "text":
                            # s = s.query("match", text=query_text)
                            s = s.query("match", text={"query":query_text, "operator":'and', "fuzziness":"auto"})
            elif search_type == "normal":
                print(f'doing normal search')
                if search_fields_any:
                    print(f'doing normal search multi match')
                    s = s.query("multi_match", query=query_text, fields=search_fields, operator="and")
                else:
                    print(f'doing normal search match all fields')
                    for f in search_fields:
                        if f == "case_name":
                            s = s.query("match", case_name={"query":query_text, "operator":'and'})
                        if f == "text":
                            s = s.query("match", text={"query":query_text, "operator":'and'})

            if highlight and "text" in search_fields:
                s = s.highlight_options(order='score', pre_tags=["**"], post_tags=["**"])
                s = s.highlight('text', fragment_size=50, number_of_fragments=5)

        if filter_author:
            s = s.filter('terms', **{'maj_opinion_author': filter_author})
        if filter_issue:
            s = s.filter('terms', **{'issue': filter_issue})
        if filter_issue_area:
            s = s.filter('terms', **{'issue_area': filter_issue_area})
        if filter_cite_id:
            s = s.filter('terms', **{'us_cite_id': filter_cite_id})
        if filter_min_votes:
            s = s.filter('range', **{'n_min_votes': filter_min_votes})
        if filter_maj_votes:
            s = s.filter('range', **{'n_maj_votes': filter_maj_votes})
        if filter_decision_direction:
            s = s.filter('terms', **{'decision_direction': filter_decision_direction})
        if filter_argument_date:
            s = s.filter('range', **{'argument_date': filter_argument_date})
        if filter_decision_date:
            s = s.filter('range', **{'decision_date': filter_decision_date})

        # sorting
        if sort_by:
            print(f'sorting by {sort_by}')
            s = s.sort(*sort_by)
            print('\n')

        # aggregations
        if agg_by:
            agg_by_str = 'by_' + agg_by
            s.aggs.bucket(agg_by_str, 'terms', field=agg_by)

        # spell suggestions
        if search_type != "wildcard":
            s = s.suggest('my_suggestion', query_text, term={'field': 'text'})

        # get the response
        response = s.execute()
        return response

def get_training_set():
    train_text, _, _, _ = get_scdb_data_raw()
    meta_data = get_scdb_metadata()

    docs = []
    for text, record in zip(train_text[:10], meta_data[:10].iterrows()):
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


def get_test_set():
    dataset = DataSetFactory.get_data(name='scdb')
    _, raw_test_corpus, _, _ = dataset.get_data(clean=False)
    _, test_meta_data = dataset.get_metadata()
    return initialize(raw_test_corpus, test_meta_data)


def initialize(corpus, metadata):
    search = DocSearch(host_ip=os.getenv('ELASTIC_HOST'), port=os.getenv('ELASTIC_PORT'), index_name='scdb')
    if search.setup(): # if no index returns true and we can create it.
        limit = -1
        docs = []
        for text, record in zip(corpus[:limit], metadata[:limit].iterrows()):
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
        search.add_docs_bulk(docs)
    return search


# if __name__ == "__main__":
#     print(os.getenv('ELASTIC_HOST'))
#     print(os.getenv('ELASTIC_PORT'))
#     print(f"elastic ping: {Elasticsearch([{'host': os.getenv('ELASTIC_HOST'), 'port': os.getenv('ELASTIC_PORT')}]).ping()}")

    # import sys
    # search = DocSearch(host_ip='172.17.0.3', port=9200, index_name='scdb')
    # # search.setup()
    # # docs = get_training_set()
    # # ret = search.add_docs_bulk(docs[:9])
    # # print(ret)
    # # print(docs[9]["_id"])
    # # # print(docs[9]["_source"])
    # # ret = search.add_doc(doc_id=docs[9]["_id"], entry=docs[9]["_source"])
    # # print(ret)
    # q = ""
    # if len(sys.argv) > 1:
    #     q = sys.argv[1]
    # print(f'\nsearch string: {q}\n')
    # res = search.search(query=q)
    # res = search.search_dsl(query=q)
    # search.search_text(query_text=q,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     wildcard=True,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     wildcard=True,
    #                     search_fields=['case_name', 'text'],
    #                     search_fields_any=False,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     wildcard=True,
    #                     search_fields=['case_name', 'text'],
    #                     search_fields_any=True,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     phrase=True,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     phrase=True,
    #                     search_fields=['text'],
    # search.search_text(query_text=q,
    #                     phrase=True,
    #                     search_fields=['case_name', 'text'],
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     phrase=True,
    #                     search_fields=['case_name'],
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     fuzzy=True,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     fuzzy=False,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     fuzzy=True,
    #                     search_fields=['text', 'case_name'],
    #                     search_fields_any=False,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     fuzzy=True,
    #                     search_fields=['case_name'],
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # print("\n==================\n")
    # search.search_text(query_text=q,
    #                     fuzzy=False,
    #                     search_fields=['case_name', 'text'],
    #                     search_fields_any=False,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     fuzzy=False,
    #                     search_fields=['case_name', 'text'],
    #                     search_fields_any=True,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}])
    # search.search_text(query_text=q,
    #                     fuzzy=False,
    #                     search_fields=['case_name', 'text'],
    #                     search_fields_any=True,
    #                     agg_by='issue_area',
    #                     sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}],
    #                     # filter_author=[105, 109],
    #                     # filter_issue=[10120, 80130],
    #                     # filter_issue_area=[10, 8],
    #                     # filter_cite_id=['540 U.S. 461'],
    #                     # filter_decision_direction=['liberal'],
    #                     # filter_min_votes={"gte": 4},
    #                     # filter_maj_votes={"lte": 8},
    #                     # filter_decision_date={"gte": 1980, "lte": 2000},
    #                     # filter_argument_date={"gte": 1980, "lte": 2000},
    #                 )

    # search.search_text_type(query_text=q,
    #                     search_fields=['case_name', 'text'],
    #                     search_fields_any=True,
    #                     # agg_by='issue_area',
    #                     # sort_by=[{'decision_date': 'desc'}, {'issue_area': 'asc'}],
    #                     # filter_author=[105, 109],
    #                     # filter_issue=[10120, 80130],
    #                     # filter_issue_area=[10, 8],
    #                     # filter_cite_id=['540 U.S. 461', '406 U.S. 164', '406 U.S. 441'],
    #                     # filter_decision_direction=['liberal'],
    #                     # filter_min_votes={"gte": 4},
    #                     # filter_maj_votes={"lte": 8},
    #                     # filter_decision_date={"gte": 1980, "lte": 2000},
    #                     # filter_argument_date={"gte": 1980, "lte": 2000},
    #                 )
