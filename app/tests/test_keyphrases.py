import random
import pprint
import spacy

from utils.dataset import ScdbData
from dms.keyphrase import get_keyphrases, compare_keyphrases


scdb = ScdbData()
scdb.download_from_textacy(overwrite=False)
scdb.prepare_data()
raw_train_corpus, raw_test_corpus, train_labels, test_labels = scdb.get_data(clean=False)
issue_area_codes = scdb.get_label_dict()
print(f"len of train set: {len(train_labels)}")
print(f"len of test set: {len(test_labels)}")

# idx = 4732, 4424
# select any one file using random number
DOC_ID = random.randint(0, len(raw_train_corpus))
text = raw_train_corpus[DOC_ID]
print(f'idx: {DOC_ID}, label: {issue_area_codes[train_labels[DOC_ID]]}')
print(f'Target:\n{raw_train_corpus[DOC_ID][:1000]}\n\n')

nlp = spacy.load("en_core_web_md")
pp = pprint.PrettyPrinter(indent=2)

print('KPs from cake')
kp = get_keyphrases(text, nlp=nlp, algo='cake')
pp.pprint(kp)
print('\n------------------------------\n')

print('KPs from textrank')
kp = get_keyphrases(text, nlp=nlp, algo='textrank')
pp.pprint(kp)
print('\n------------------------------\n')

print('KPs from yake')
kp = get_keyphrases(text, nlp=nlp, algo='yake')
pp.pprint(kp)
print('\n------------------------------\n')

print('KPs from pytextrank')
kp = get_keyphrases(text, nlp=nlp, algo='pytextrank')
pp.pprint(kp)
print('\n------------------------------\n')

print('Comparing keyphrases')
_, formatted = compare_keyphrases(text, nlp=nlp)
print(formatted)

# test
print('Testing no text input')
kp = get_keyphrases(None, nlp=nlp, algo='cake')
print(kp)
print()

print('Testing unknown algo')
kp = get_keyphrases("How are you!", nlp=nlp, algo='fake')
print(kp)
print()

print('Testing short text')
kp = get_keyphrases("How are you!", nlp=nlp, algo='cake')
print(kp)
print()

print('Testing top n = 0')
kp = get_keyphrases("How are you!", nlp=nlp, algo='cake', TOP_N=0)
print(kp)
print()

print('Testing large top n and short text')
kp = get_keyphrases("How are you!", nlp=nlp, algo='cake', TOP_N=10000)
print(kp)
print()

print('Testing large top n and long text')
kp = get_keyphrases(text, nlp=nlp, algo='cake', TOP_N=10000)
print(kp)
print()
