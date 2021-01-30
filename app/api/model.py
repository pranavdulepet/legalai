from datetime import datetime as dt
from typing import List, Tuple, Optional
import spacy
from spacy import displacy

from utils.dataset import DataSetFactory
from utils.clean_text import clean_text
from api.types import ClfResponse, KpAlgos, KpAlgosTypes
from api.types import SimTopics, SimDocs, SimResponse, SimTypes
from dms.classify import Classify
from dms.similar import Similarity, get_supported_sim_types
from dms.ner import Ner
from dms.keyphrase import get_keyphrases, get_supported_keyphrase_algos
from dms.summary import get_summary
from search import initialize
# from database import add_search_history, read_search_history

# TODO: instead of return True/False, raise Exceptions.
class API:
    def __init__(self,
                ds_name: str = 'scdb',
                clf: bool = True,
                # sim: SimTypes = SimTypes.nmf,
                sim_lda: bool = True,
                sim_nmf: bool = True,
                ner: bool = True,
                kp: bool = True,
                summ: bool = True,
                search: bool = True) -> None:
        '''
        Set a particular module to True to load it.
        * ds_name - name of the dataset to be loaded. This will be done by default.
        * clf: bool - classifier
        * sim - "lda" or "nmf"
        * ner: bool - named entity recognizer
        * kp: bool - keyphrases
        * summ: bool - extractive summarizer
        '''
        self._load_dataset(ds_name)
        if clf: self._load_classifier(ds_name=ds_name)
        if sim_lda: self._load_sim_lda(ds_name=ds_name)
        if sim_nmf: self._load_sim_nmf(ds_name=ds_name)
        if ner or kp or summ: self._load_nlp()
        if ner: self._load_ner()
        if search: self._load_search()

    def _load_dataset(self, ds_name:str) -> bool:
        ''' Prepare dataset files, returns false if loading fails '''
        # TODO: return false if loading fails.
        self.dataset = DataSetFactory.get_data(name=ds_name)
        self.dataset.prepare_data()
        self.clean_train_corpus, self.clean_test_corpus, self.train_labels, self.test_labels = self.dataset.get_data(clean=True)
        self.raw_train_corpus, self.raw_test_corpus, _, _ = self.dataset.get_data(clean=False)
        self.issue_area_codes = self.dataset.get_label_dict()

        _, self.test_meta_data = self.dataset.get_metadata()
        print(f"len of train set: {len(self.train_labels)}")
        print(f"len of test set: {len(self.test_labels)}")
        return True

    def _load_classifier(self, ds_name:str) -> bool:
        ''' load the classifier model, returns false, if loading fails. '''
        # TODO: return false if loading fails.
        start_time = dt.now()
        self.clf = Classify(ds_name=ds_name)
        self.clf.load()
        print(f"Loading classifier took: {dt.now()-start_time}")
        return True

    def _load_sim_lda(self, ds_name:str) -> bool:
        ''' loads the lda model for similarity check. returns false if loading fails '''
        # TODO: return false if loading fails.
        start_time = dt.now()
        self.sim_lda = Similarity(model_type='lda', ds_name=ds_name)
        self.sim_lda.load()
        self.sim_lda.set_corpus(self.clean_train_corpus)
        print(f"Loading lda took: {dt.now()-start_time}")
        return True

    def _load_sim_nmf(self, ds_name:str) -> bool:
        ''' loads the nmf model for similarity check. returns false if loading fails '''
        # TODO: return false if loading fails.
        start_time = dt.now()
        self.sim_nmf = Similarity(model_type='nmf', ds_name=ds_name)
        self.sim_nmf.load()
        self.sim_nmf.set_corpus(self.clean_train_corpus)
        print(f"Loading nmf took: {dt.now()-start_time}")
        return True

    def _load_nlp(self):
        ''' Load the spacy nlp model. This is needed if either of
        ner, kp or summarizer is loaded
        '''
        self.nlp = spacy.load("en_core_web_md")

    def _load_ner(self):
        ''' Load the Named Entity Recognizer '''
        self.ner = Ner(self.nlp)

    def _load_search(self):
        ''' Init the elastic search indexes and add documents '''
        self.search = initialize(self.raw_test_corpus, self.test_meta_data)

    def get_cls_label(self, doc: str) -> ClfResponse:
        '''
        Takes a document number as input and returns the Classification Prediction.

        Response includes:

            doc_num: int, input document number

            text: str, the truncated text of the document

            actual_label_num: int, the actual label number

            actual_label_str: str, the actual label string

            pred_label_num: int, the predicted label number

            pred_label_str: str, the predicted label string

        '''
        label_pred_num  = self.clf.predict(clean_text(doc))
        label_pred_str = self.issue_area_codes[label_pred_num]
        clfresp = ClfResponse(
                            pred_label_num = label_pred_num,
                            pred_label_str = label_pred_str
                        )
        return clfresp

    def get_topics_lda(self) -> SimTopics:
        return self.get_topics(self.sim_lda)

    def get_topics_nmf(self) -> SimTopics:
        return self.get_topics(self.sim_nmf)

    # Document Similarity
    def get_topics(self, sim_model:Similarity) -> SimTopics:
        ''' Returns a List of lda/nmf topics and 10 words in each topic. '''
        topic_words = sim_model.topic_words()
        topic_str = []
        for i, top in enumerate(topic_words):
            topic_str.append(" ".join(top))
        return SimTopics(topics=topic_str)

    def get_similar_documents_lda(self, doc:str, corpus: List[str], count:int = 5, force=False) -> SimResponse:
        return self.get_similar_documents(sim_model=self.sim_lda, doc=doc, count=count, corpus=corpus, force=force)

    def get_similar_documents_nmf(self, doc:str, corpus: List[str], count:int = 5, force=False) -> SimResponse:
        return self.get_similar_documents(sim_model=self.sim_nmf, doc=doc, count=count, corpus=corpus, force=force)

    def get_similar_documents(self, sim_model:Similarity, doc:str, corpus: List[str], count:int=5, force=False
    ) -> SimResponse:
        ''' Returns a set of similar documents to the query document '''
        sims, idxs, doc_topics, corpus_topics = \
            sim_model.get_similar_documents(
                doc,
                # self.clean_train_corpus,
                corpus,
                count=count,
                force=force
            )

        # doc_attr = {}
        # doc_attr["doc_topics"] = doc_topics

        # TODO: Change corpus to users own
        sim_docs_list = []
        for sim, idx, corpus_topic in zip(sims, idxs, corpus_topics):
            sim_doc_attr = {}
            # TODO: replace doc_num with the hyperlink of the document.
            sim_doc_attr["doc_num"] = idx
            sim_doc_attr["text"] = self.raw_train_corpus[idx][:1000]
            sim_doc_attr["actual_label_num"] = self.train_labels[idx]
            sim_doc_attr["actual_label_str"] = self.issue_area_codes[self.train_labels[idx]]
            sim_doc_attr["doc_topics"] = corpus_topic

            sim_docs = {}
            sim_docs["doc"] = sim_doc_attr
            sim_docs["score"] = sim

            sim_docs_list.append(sim_docs)

        sim_resp = {}
        sim_resp["doc"] = doc_topics
        sim_resp["sim_docs_list"] = sim_docs_list
        sim_resp = SimResponse(**sim_resp)
        return sim_resp

    # Named Entities
    def get_entities(self,
                    doc: str,
                    filter_labels: List[str] = Ner.NER_LABELS,
                    filter_attrs: List[str] = Ner.NER_ATTRS,
                    ) -> List[List[str]]:
        '''
        Returns a list of named entities from the document.

        The entities can be filtered by the labels and only the attributes
        listed will be returned.

        Return will be a list. Each item in the list will in turn contain
        the attributes specified.
        '''
        ners = self.ner.get_named_entities(doc,
                                    filter_labels=filter_labels,
                                    filter_attrs=filter_attrs)
        return ners

    def get_entities_html(self,
                    doc: str,
                    filter_labels: List[str] = Ner.NER_LABELS,
                    ) -> List[List[str]]:
        '''
        Returns a list of named entities from the document.

        The entities can be filtered by the labels and only the attributes
        listed will be returned.

        Return will be a list. Each item in the list will in turn contain
        the attributes specified.
        '''
        ners = self.ner.get_named_entities_html(doc,
                                    filter_labels=filter_labels)
        return ners

    def get_highlights_html(self,
                    doc: str, TOP_N:int=5
        ):
        '''
        Returns the full document with highlights.

        '''
        summary = get_summary(doc, self.nlp, TOP_N=TOP_N)
        ents = []
        for summ in summary:
            _, start, end = summ
            ents.append({"start": start, "end": end, "label": ""})
        ex = [{
            "text": doc,
            "ents": ents,
            "title": None
            }]
        html = displacy.render(ex, style="ent", manual=True, page=True)
        return html

    def get_entity_labels_attrs(self):
        ''' Return the full list of available entity labels and attributes '''
        return self.ner.get_entity_labels(), self.ner.get_entity_attrs()

    def get_keyphrase_algos(self):
        return get_supported_keyphrase_algos()

    def get_sim_types(self):
        return get_supported_sim_types()

    # keyphrases
    def keyphrases(self, doc: str,
                    algo: KpAlgos = KpAlgosTypes.cake,
                    TOP_N: int = 10
                    ) -> List[str]:
        '''
        Return the list of keyphrases from the document.

        The algorithm can be any of 'cake', 'yake', 'textrank', 'pytextrank'

        TOP_N: int - specify the number of keyphrases to return

        Return value is a List of strings.

        '''
        kp = get_keyphrases(doc, self.nlp, algo=algo, TOP_N=TOP_N)
        return kp

    def summary(self, doc: str, TOP_N:int=5) -> List[str]:
        '''
        Returns the extractive summary of the document.

        Can specify the number of sentences using TOP_N, default=5.

        '''
        summary = get_summary(doc, self.nlp, TOP_N=TOP_N)
        return summary

    def query(self, query_text: str) -> List[str]:
        '''
        Search for a text in the corpus
        '''
        # add_search_history(query_text)
        response = self.search.search_text(query_text=query_text)
        # TODO: return a standard json response
        print(response.hits.total.value)
        results = []
        for hit in response:
            for frag in hit.meta.highlight.text:
                results.append(frag)
        return results

    # def get_search_history(self) -> List[str]:
    #     '''
    #     Returns the history of search texts.
    #     '''
    #     return read_search_history()
