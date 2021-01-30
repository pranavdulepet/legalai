import re
import string
import nltk
import unicodedata
import glob
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def clean_text(t):
    t = t.lower()
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    t = re.compile(r"<([^>]+)>", re.UNICODE).sub("", t)
    t = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE).sub(" ", t)
    t = re.compile(r"(\s)+", re.UNICODE).sub(" ", t)
    t = re.compile(r"[0-9]+", re.UNICODE).sub("", t)
    t = " ".join(e for e in t.split() if len(e) >= 3)
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.append('footnote')
    t = " ".join(ps.stem(word) for word in t.split() if word not in stopword_list)
    return t

def read_file(path):
    with open(path, 'r') as fin:
        return fin.read()

def read_files(file_names):
    return [read_file(fname) for fname in file_names]

if __name__ == "__main__":
    t = "<i>Hello</i> <b>World</b>! Sómě Áccěntěd těxt.   Plus some23 num34bers     007. if-you#can%read$this&then@this#method^works           . Text, texting, texted, texter"
    print(t)
    t = clean_text(t)
    print(t)
