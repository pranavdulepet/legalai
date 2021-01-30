import random

from dms.similar import Similarity
from utils.dataset import ScdbData


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
    topic_words = scdb_lda.topic_words()
    for i, top in enumerate(topic_words):
        msg = "Topic " + str(i) + ": " + " ".join(top)
        print(msg)
    sims, idxs, doc_topics, corpus_topics = scdb_lda.get_similar_documents(test_sample, clean_train_corpus, k=5)

    print('\n************LDA***************\n')
    print(f"Top 3 topics in this doc: {doc_topics}")
    print('\n------Similar Documents-----\n')
    for sim, idx, corpus_topic in zip(sims, idxs, corpus_topics):
        print(f'idx: {idx}, label: {issue_area_codes[train_labels[idx]]}, cos sim: {sim}')
        print(f'Topics: {corpus_topic}')
        print(raw_train_corpus[idx][:1000])
        print('\n-----------\n')

    print('\n-----------------------------------\n')

    # nmf
    scdb_nmf = Similarity(model_type="nmf")
    topic_words = scdb_nmf.topic_words()
    for i, top in enumerate(topic_words):
        msg = "Topic " + str(i) + ": " + " ".join(top)
        print(msg)
    sims, idxs, doc_topics, corpus_topics = scdb_nmf.get_similar_documents(test_sample, clean_train_corpus, k=5)

    print('\n************NMF***************\n')
    print(f"Top 3 topics in this doc: {doc_topics}")
    print('\n------Similar Documents-----\n')
    for sim, idx, corpus_topic in zip(sims, idxs, corpus_topics):
        print(f'idx: {idx}, label: {issue_area_codes[train_labels[idx]]}, cos sim: {sim}')
        print(f'Topics: {corpus_topic}')
        print(raw_train_corpus[idx][:1000])
        print('\n-----------\n')

get_sim_docs(DOC_ID=None)  # 432, 799
