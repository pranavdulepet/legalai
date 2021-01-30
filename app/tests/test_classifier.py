import random

from utils.dataset import ScdbData
from dms.classify import Classify


scdb = ScdbData()
scdb.download_from_textacy(overwrite=False)
scdb.prepare_data()
clean_train_corpus, clean_test_corpus, train_labels, test_labels = scdb.get_data(clean=True)
print(f"len of train set: {len(train_labels)}")
print(f"len of test set: {len(test_labels)}")

# # the classifier
# scdb_clf = Classify()

# # Train and Test
# print(f"Training the classifier")
# scdb_clf.train(clean_train_corpus, train_labels)
# print(f"Testing the classifier")
# scdb_clf.test(clean_test_corpus, test_labels)

# predict
scdb_clf = Classify()
scdb_clf.load()
issue_area_codes = scdb.get_label_dict()

# get some random test document
len_samples = len(test_labels)
doc_list = random.sample(range(0, len_samples), 5)

# predict few samples.
for doc_num in doc_list:
    doc = clean_test_corpus[doc_num]
    label_actual_num = test_labels[doc_num]
    label_actual_str = issue_area_codes[label_actual_num]
    label_pred_num  = scdb_clf.predict(doc)
    label_pred_str = issue_area_codes[label_pred_num]
    print(f"Doc# {doc_num}, Actual: {(label_actual_num, label_actual_str)}, "\
            + f"Predicted: {(label_pred_num, label_pred_str)}")
    print(doc[:1000])
    print('\n-----------------\n')