import json

from utils.dataset import ScdbData

# TODO: convert to pytest

scdb = ScdbData()
scdb.download_from_textacy(overwrite=False)
scdb.prepare_data()
train_corpus, test_corpus, train_labels, test_labels = scdb.get_data(clean=True)

print("Sample train corpus")
for text in train_corpus[:5]:
    print(text[:100])
    print('-'*10)
print(train_labels[:5])
print()
print("Sample test corpus")
for text in test_corpus[:5]:
    print(text[:100])
    print('-'*10)
print(train_labels[:5])
print('\n-----------------\n')

# Metadata
train_meta, test_meta = scdb.get_metadata()
print('Train Metadata head')
print(train_meta.head())
print('Train Metadata head')
print(test_meta.head())
print('\n-----------------\n')

doc_num = 0
print(f"Metadata of document number {doc_num} from train corpus")
train_doc_meta = scdb.get_doc_metadata(doc_num=doc_num, train=True)
print(train_doc_meta)
print(f"Metadata of document number {doc_num} from test corpus")
train_doc_meta = scdb.get_doc_metadata(doc_num=doc_num, train=False)
print(train_doc_meta)

print('Dataset info')
info = scdb.get_info()
print(json.dumps(info, indent=2))

print('Label dictionary')
label_map = scdb.get_label_dict()
print(label_map)
print(json.dumps(label_map, indent=2))

label_num = 2
print('Label string of label num {label_num}')
label = scdb.get_label_str(num=label_num)
print(label)
