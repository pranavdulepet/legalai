import random
from wasabi import table

from utils.dataset import ScdbData
from dms.ner import Ner


scdb = ScdbData()
scdb.download_from_textacy(overwrite=False)
scdb.prepare_data()
raw_train_corpus, raw_test_corpus, train_labels, test_labels = scdb.get_data(clean=False)
issue_area_codes = scdb.get_label_dict()
print(f"len of train set: {len(train_labels)}")
print(f"len of test set: {len(test_labels)}")

DOC_ID = random.randint(0, len(raw_train_corpus))
# DOC_ID = 587
text = raw_train_corpus[DOC_ID][:2000]
print(f'idx: {DOC_ID}, label: {issue_area_codes[train_labels[DOC_ID]]}')
print(f'Target:\n{raw_train_corpus[DOC_ID][:1000]}\n\n')

ner = Ner()
filter_labels = ['LAW', 'ORG', 'GPE']
filter_attrs = ['text', 'label_', 'start', 'end']
ners = ner.get_named_entities(text, filter_labels=filter_labels, filter_attrs=filter_attrs)

# header = ("Text", "Label", "start", "end")
header = filter_attrs
widths = (35, 10, 6, 6)
aligns = ("l", "l", "l", "l")

formatted = table(ners, header=header, divider=True, widths=widths, aligns=aligns)
print(formatted)
