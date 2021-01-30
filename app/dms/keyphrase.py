from textacy import ke
import spacy
import random
import pprint
import pytextrank
from wasabi import table
import numpy as np

from api.types import KpAlgosTypes


def get_supported_keyphrase_algos():
    kp_algo_list = [
        KpAlgosTypes.cake,
        KpAlgosTypes.yake,
        KpAlgosTypes.textrank,
        KpAlgosTypes.pytextrank
        ]
    return kp_algo_list


def get_keyphrases(text, nlp=None, algo='cake', TOP_N=50):
    """
        TODO:
        The nlp object can be initialized at one location.
        Since it is needed for keyphrase extraction, ner and
        extractive summarization.
    """
    """
        TODO:
        The quality of keyphrases can be improved by removing near duplicates.
        Sometimes, there are keyphrases like:
        1. Missouri court and Missouri Supreme Court
        2. incredible explanation and prosecutorial explanation

        These are from index 2902 of the train set and using the textacy-textrank
        algorithm.

        Here, from 1 and 2 we can filter out any one keyphrase, since the other
        one does not provide much extra value.

        This could be done by looking for similar words in the extracted keyphrases
        and removing near duplicates etc.
        Or there could be other methods, we need to check.
    """
    if text == None or text == "":
        print("No text input provided")
        return None

    if TOP_N == 0:
        print("Please select TOP_N>0")
        return None

    if not nlp:
        nlp = spacy.load("en_core_web_md")

    # get all the keyphrases from the document
    doc = nlp(text)

    if algo == 'textrank':
        tr = ke.textrank(doc, topn=TOP_N)
        kp = [t[0] for t in tr]
    elif algo == 'cake':
        cake = ke.scake(doc, topn=TOP_N)
        kp = [t[0] for t in cake]
    elif algo == 'yake':
        yake = ke.yake(doc, topn=TOP_N)
        kp = [t[0] for t in yake]
    elif algo == 'pytextrank':
        pytr = pytextrank.TextRank()
        if "textrank" not in nlp.pipe_names:
            nlp.add_pipe(pytr.PipelineComponent, name="textrank", last=True)
        doc = nlp(text)
        kp = doc._.phrases[:TOP_N]
        kp = [p.text for p in kp]
    else:
        print("Unknown algorithm")
        return None

    return kp

def compare_keyphrases(text, nlp=None, TOP_N=50):
    if text == None or text == "":
        print("No text input provided")

    if TOP_N == 0:
        print("Please select TOP_N>0")

    header = ("textrank", "cake", "pytextrank")
    widths = (30, 30, 30)
    aligns = ("l", "l", "l")

    if not nlp:
        nlp = spacy.load("en_core_web_md")

    # get all the keyphrases from the document
    doc = nlp(text)

    tr = ke.textrank(doc, topn=TOP_N)
    tr = np.asarray([t[0] for t in tr])
    tr = tr.reshape(-1, 1)

    cake = ke.scake(doc, topn=TOP_N)
    cake = np.asarray([t[0] for t in cake])
    cake = cake.reshape(-1, 1)

    # the results of yake are not so good. not using for comparison.
    yake = ke.yake(doc, topn=TOP_N)
    yake = np.asarray([t[0] for t in yake])
    yake = yake.reshape(-1, 1)

    pytr = pytextrank.TextRank()
    if "textrank" not in nlp.pipe_names:
        nlp.add_pipe(pytr.PipelineComponent, name="textrank", last=True)
    doc = nlp(text)
    phrases = doc._.phrases[:TOP_N]
    pytr = np.asarray([p.text for p in phrases])
    pytr = pytr.reshape(-1, 1)

    kp = np.hstack((tr, cake, pytr))
    formatted = table(kp, header=header, divider=True, widths=widths, aligns=aligns)
    # print(formatted)
    return kp, formatted
