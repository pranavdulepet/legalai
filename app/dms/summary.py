from textacy import ke
import spacy
import random
import pprint
import pytextrank
from wasabi import wrap
import numpy as np


def get_summary(text, nlp=None, TOP_N=5):
    """
        For now, only extractive summary is implemented.
        Abstractive summary could be better but difficult to implement for
        long document summarization.
        1.
        Few ideas are to do extractive summary first and then do abstractive
        summary on top of that.
        2.
        Other would be to divide document by parts and then summarize each path.
        This would apply equally to both Extractive and Abstractive summaries.
        3.
        Also need to do summarization in the form of begin-body-conclusion.
        This could also be customized to a specific domain.
    """
    """
        TODO:
        The nlp object can be initialized at one location.
        Since it is needed for keyphrase extraction, ner and
        extractive summarization.
    """
    if text == None or text == "":
        print("No text input provided")
        return None

    if TOP_N == 0:
        print("Please select TOP_N>0")
        return None

    if not nlp:
        print(f'Loading nlp model')
        nlp = spacy.load("en_core_web_md")
    # nlp = en_core_web_md.load()

    pytr = pytextrank.TextRank()
    if "textrank" not in nlp.pipe_names:
        nlp.add_pipe(pytr.PipelineComponent, name="textrank", last=True)
    doc = nlp(text)
    summary = doc._.textrank.summary(limit_phrases=15, limit_sentences=TOP_N)
    sents = []
    for sent in summary:
        sents.append((sent.text, sent.start_char, sent.end_char))
        # print(dir(sent))
        # print(sent.start_char, sent.end_char)
    return sents
    # TODO: instead of returning span, return the actual list of strings.
    # TODO: also return the start and end of sentences, so that it can be highlighted.
