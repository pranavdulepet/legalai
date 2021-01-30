from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import random
import os.path
from datetime import datetime as dt
import pathlib

from utils.dataset import ScdbData

class Similarity:

    def __init__(self, model_type='lda', model_path='model/', ds_name="scdb",
                n_features=1000, n_components=30, n_top_words=10):
        if model_path and model_path != "":
            self.MODEL_PATH = model_path
        if ds_name and ds_name != "":
            self.DS_NAME = ds_name
        if model_type and model_type != "":
            self.MODEL_TYPE = model_type
        self.n_features = n_features
        self.n_components = n_components
        self.n_top_words = n_top_words
        self.model = None
        self.matrix = None

    def train(self, train_corpus):
        if self.MODEL_TYPE == "lda":
            return self.train_lda_model(train_corpus)
        elif self.MODEL_TYPE == "nmf":
            return self.train_nmf_model(train_corpus)

    def test(self, test_corpus):
        if self.MODEL_TYPE == "lda":
            return self.test_lda_model(test_corpus)
        elif self.MODEL_TYPE == "nmf":
            return self.test_nmf_model(test_corpus)

    def train_lda_model(self, train_corpus):
        print(f'Training LDA model')
        start_time = dt.now()
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=self.n_features)

        lda_tf = LatentDirichletAllocation(n_components=self.n_components,
                                        max_iter=25,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        model = Pipeline([
            ('tf_vectorizer', tf_vectorizer),
            (self.MODEL_TYPE, lda_tf)
            ])
        model.fit(train_corpus)

        lda_dump_path = self.MODEL_PATH + self.DS_NAME + "_" + self.MODEL_TYPE + ".joblib"
        pathlib.Path(self.MODEL_PATH).mkdir(parents=True, exist_ok=True)
        print(f"saving lda model to {lda_dump_path}")
        joblib.dump(model, lda_dump_path)

        tf_feature_names = tf_vectorizer.get_feature_names()
        self.top_topic_words = self._get_topic_words(lda_tf, tf_feature_names, self.n_top_words)
        print(f"Training lda model took {dt.now() - start_time}")
        return self.top_topic_words

    def test_lda_model(self, test_corpus):
        print(f'Testing LDA model')

        lda_dump_path = self.MODEL_PATH + self.DS_NAME + "_" + self.MODEL_TYPE + ".joblib"
        if not os.path.isfile(lda_dump_path):
            print(f"Trained model not found at paths {lda_dump_path}")

        print(f'Testing LDA model: loading trained model from paths {lda_dump_path}')
        model = joblib.load(lda_dump_path)

        print(f'Testing LDA model: splitting corpus into 2 halves')
        articles_1 = [text[:len(text)//2] for text in test_corpus]
        articles_2 = [text[len(text)//2:] for text in test_corpus]

        lda_1 = model.transform(articles_1)
        lda_2 = model.transform(articles_2)

        print(f'Testing LDA model: calculating cosine similarities')
        cos_sim = cosine_similarity(lda_1, lda_2)
        cos_sim_diag = cos_sim.diagonal()

        print(f'Testing LDA model: calculating intra and inter similarities')
        intra_sim = np.mean(cos_sim_diag)
        inter_sim = np.mean(cos_sim)

        print(f"The intra similarity is {intra_sim} and inter similarity is {inter_sim}")
        print(f"Intra similarity should be higher and inter similarity should be lower\n\n")
        return intra_sim, inter_sim

    def train_nmf_model(self, train_corpus):
        print(f'Training NMF model')
        start_time = dt.now()
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=self.n_features)

        tfidf_vectorizer = TfidfTransformer(smooth_idf=True, use_idf=True)

        nmf_tfidf = NMF(n_components=self.n_components, random_state=1,
                        alpha=.1, l1_ratio=.5)

        model = Pipeline([
            ('tf_vectorizer', tf_vectorizer),
            ('tfidf_vectorizer', tfidf_vectorizer),
            (self.MODEL_TYPE, nmf_tfidf)
            ])

        print(f'Training NMF model: fitting nmf model on tfidf')
        model.fit(train_corpus)

        nmf_dump_path = self.MODEL_PATH + self.DS_NAME + "_" + self.MODEL_TYPE + ".joblib"
        pathlib.Path(self.MODEL_PATH).mkdir(parents=True, exist_ok=True)
        print(f"saving nmf model to {nmf_dump_path}")
        joblib.dump(model, nmf_dump_path)

        tf_feature_names = tf_vectorizer.get_feature_names()
        self.top_topic_words = self._get_topic_words(nmf_tfidf, tf_feature_names, self.n_top_words)
        print(f"Training nmf model took {dt.now() - start_time}")
        return self.top_topic_words

    def test_nmf_model(self, test_corpus):
        print(f"Testing NMF model")
        nmf_dump_path = self.MODEL_PATH + self.DS_NAME + "_" + self.MODEL_TYPE + ".joblib"
        if not os.path.isfile(nmf_dump_path):
            print(f"Trained model not found at paths {nmf_dump_path}")

        print(f"Testing NMF model: loading models")
        model = joblib.load(nmf_dump_path)

        print(f"Testing NMF model: splitting corpus into 2 halves")
        articles_1 = [text[:len(text)//2] for text in test_corpus]
        articles_2 = [text[len(text)//2:] for text in test_corpus]

        nmf_1 = model.transform(articles_1)
        nmf_2 = model.transform(articles_2)

        print(f"Testing NMF model: calculating cosine similarity")
        cos_sim = cosine_similarity(nmf_1, nmf_2)
        cos_sim_diag = cos_sim.diagonal()

        print(f"Testing NMF model: calculating intra and inter similarity")
        intra_sim = np.mean(cos_sim_diag)
        inter_sim = np.mean(cos_sim)

        print(f"The intra similarity is {intra_sim} and inter similarity is {inter_sim}")
        print(f"Intra similarity should be higher and inter similarity should be lower")
        return intra_sim, inter_sim

    def load(self):
        # load the model and keep ready for analysis.
        if self.model != None:
            return 0

        print(f"Loading {self.MODEL_TYPE} model")
        model_dump_path = self.MODEL_PATH + self.DS_NAME + "_" + self.MODEL_TYPE + ".joblib"
        if not os.path.isfile(model_dump_path):
            print(f"Trained model not found at path {model_dump_path}")
            return -1

        print(f'Loading from {model_dump_path}')
        self.model = joblib.load(model_dump_path)

        tf_feature_names = self.model['tf_vectorizer'].get_feature_names()
        print(f'Getting topic words')
        self.top_topic_words = self._get_topic_words(self.model[self.MODEL_TYPE], tf_feature_names, self.n_top_words)
        return 0

    def set_corpus(self, corpus, force=False):
        if not force:
            if not (self.matrix is None):
                print(f"Corpus already created")
                return 0

        if corpus == None or len(corpus) == 0:
            print(f"No corpus provided")
            return -1

        if self.load() == -1:
            print(f"No model found during load")
            return -1

        print(f'Transforming to {self.MODEL_TYPE} matrix')
        self.matrix = self.model.transform(corpus)
        return 0

    def add_docs(self, docs):
        # TODO: append this vector to self.matrix
        pass

    def remove_docs(self, doc_ids):
        # TODO: remove the corresponding vectors from self.matrix
        pass

    def _get_most_similar_vectors(self, query, matrix, count=5):
        if len(query.shape) == 1 or query.shape[0] == 1:
            query = query.reshape(1, -1)
        else:
            print(f'Takes only one vector for comparison, actual shape: {query.shape}')
            return None

        sims = cosine_similarity(query, matrix)[0] # list of cosine similarities
        idxs = sims.argsort()[-count:][::-1]
        return sims[idxs], idxs # the top c positional index of the largest cosine similarities

    def get_similar_documents(self, doc, corpus, count=10, force=False):
        """ get a text and return most similar documents. """
        # preload the model.
        print(f'get_simimar_documents: loading models')
        if self.load() == -1:
            print(f"No model found during load")
            return None

        print(f'get_simimar_documents: setting corpus')
        if self.set_corpus(corpus, force) == -1:
            print(f"No corpus provided")
            return None

        print(f'get_simimar_documents: {self.MODEL_TYPE} transform')
        query = self.model.transform([doc])

        print(f'get_simimar_documents: Getting similar vectors')
        sims, idxs = self._get_most_similar_vectors(query, self.matrix, count)

        print(f'get_simimar_documents: Getting top topics of similar documents')
        corpus_topics = [self._get_top_topics(self.matrix[i]) for i in idxs]

        print(f'get_simimar_documents: Getting document topics')
        doc_topics = self._get_top_topics(query[0])

        return sims, idxs, doc_topics, corpus_topics

    def _get_topic_words(self, model, feature_names, n_top_words):
        top_topic_words = []
        for topic_idx, topic in enumerate(model.components_):
            topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            top_topic_words.append(topic_words)
        return top_topic_words

    def _get_top_topics(self, x):
        n = normalize(x[:,np.newaxis], axis=0, norm='l1').ravel()
        idxs = n.argsort()[-3:][::-1]
        return list(zip(idxs, np.around(n[idxs], 2)))

    def topic_words(self):
        if self.load() == -1:
            return -1
        return self._get_topic_words(self.model[self.MODEL_TYPE],
            self.model['tf_vectorizer'].get_feature_names(), self.n_top_words)

def get_sim_docs(DOC_ID=None):
    # run this function any number of times to see different outputs.
    print(f"Getting random document")
    scdb = ScdbData()
    scdb.download_from_textacy(overwrite=False)
    scdb.prepare_data()
    clean_train_corpus, clean_test_corpus, train_labels, test_labels = scdb.get_data(clean=True)
    raw_train_corpus, raw_test_corpus, _, _ = scdb.get_data(clean=False)
    print(f"len of train set: {len(train_labels)}")
    print(f"len of test set: {len(test_labels)}")
    issue_area_codes = scdb.get_label_dict()

    if not DOC_ID:
        DOC_ID = random.randint(0, len(test_labels))
    raw_test_sample = raw_test_corpus[DOC_ID]
    test_sample = clean_test_corpus[DOC_ID]
    print(f'Target: idx: {DOC_ID}, label: {issue_area_codes[test_labels[DOC_ID]]}')
    print('\n-------------\n')
    print(f'{raw_test_sample[:1000]}')
    print('\n-------------\n')

    # lda
    scdb_lda = Similarity(model_type="lda")
    # sims, idxs, doc_topics, corpus_topics = scdb_lda.get_similar_documents(test_sample, raw_train_corpus[:100], c=10)
    sims, idxs, doc_topics, corpus_topics = scdb_lda.get_similar_documents(test_sample, clean_train_corpus, c=5)

    print('\n************LDA***************\n')
    print(f"Top 3 topics in this doc: {doc_topics}")
    print('\n------Similar Documents-----\n')
    for sim, idx, corpus_topic in zip(sims, idxs, corpus_topics):
        print(f'idx: {idx}, label: {issue_area_codes[train_labels[idx]]}, cos sim: {sim}')
        print(f'Topics: {corpus_topic}')
        print(raw_train_corpus[idx][:1000])
        print('\n-----------\n')

    # nmf
    scdb_nmf = Similarity(model_type="nmf")
    sims, idxs, doc_topics, corpus_topics = scdb_nmf.get_similar_documents(test_sample, clean_train_corpus, c=5)

    print('\n************NMF***************\n')
    print(f"Top 3 topics in this doc: {doc_topics}")
    print('\n------Similar Documents-----\n')
    for sim, idx, corpus_topic in zip(sims, idxs, corpus_topics):
        print(f'idx: {idx}, label: {issue_area_codes[train_labels[idx]]}, cos sim: {sim}')
        print(f'Topics: {corpus_topic}')
        print(raw_train_corpus[idx][:1000])
        print('\n-----------\n')


def get_supported_sim_types():
    return ['lda', 'nmf']


if __name__ == "__main__":
    # index 412, 1086
    # get the corpus
    scdb = ScdbData()
    scdb.prepare_data()
    clean_train_corpus, clean_test_corpus, train_labels, test_labels = scdb.get_data(clean=True)

    # init lda model
    lda_sim_scdb = Similarity(model_type='lda')

    # train
    lda_sim_scdb.train(clean_train_corpus)

    # test
    intra_sim, inter_sim = lda_sim_scdb.test(clean_test_corpus)
    print(f"lda intra and inter sims {intra_sim}, {inter_sim}")

    # init nmf model
    nmf_sim_scdb = Similarity(model_type='nmf')

    # train
    nmf_sim_scdb.train(clean_train_corpus)

    # test
    intra_sim, inter_sim = nmf_sim_scdb.test(clean_test_corpus)
    print(f"nmf intra and inter sims {intra_sim}, {inter_sim}")

    lda_sim_scdb = Similarity(model_type='lda')
    topic_words = lda_sim_scdb.topic_words()
    for i, top in enumerate(topic_words):
        msg = "Topic " + str(i) + ": " + " ".join(top)
        print(msg)

    print('\n-----------------------------------\n')
    nmf_sim_scdb = Similarity(model_type='nmf')
    topic_words = nmf_sim_scdb.topic_words()
    for i, top in enumerate(topic_words):
        msg = "Topic " + str(i) + ": " + " ".join(top)
        print(msg)

    get_sim_docs(DOC_ID=None)  # 432, 799
