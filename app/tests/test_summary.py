import random
import spacy
from wasabi import wrap

from utils.dataset import ScdbData
from dms.summary import get_summary

scdb = ScdbData()
scdb.download_from_textacy(overwrite=False)
scdb.prepare_data()
raw_train_corpus, raw_test_corpus, train_labels, test_labels = scdb.get_data(clean=False)
issue_area_codes = scdb.get_label_dict()
print(f"len of train set: {len(train_labels)}")
print(f"len of test set: {len(test_labels)}")

# select any one file using random number
DOC_ID = random.randint(0, len(raw_train_corpus))
text = raw_train_corpus[DOC_ID]
print(f'idx: {DOC_ID}, label: {issue_area_codes[train_labels[DOC_ID]]}')
print(f'Target:\n{raw_train_corpus[DOC_ID][:1000]}\n\n')

print('Loading spacy model')
nlp = spacy.load("en_core_web_md")

print('Testing n=5')
summary = get_summary(text, nlp, TOP_N=5)
for sent in summary:
    wrapped = wrap(sent, indent=0)
    print(wrapped)
    print('\n--------------\n')

# test
print('Testing no text input')
summ = get_summary(None, nlp, TOP_N=5)
if summ: print(list(summ))
print('*'*80)

# FIXME: this is giving div by zero error.
# summ = get_summary("How are you!")
# if summ: [print(s) for s in summ]
# print('*'*80)

print('Testing n=0')
summ = get_summary(text, nlp, TOP_N=0)
if summ: print(list(summ))
print('*'*80)

print('Testing large n=10000')
summ = get_summary(text, nlp, TOP_N=10000)
if summ: print(list(summ))
print('*'*80)
