import glob
import pandas as pd
import pathlib
import json
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import textacy.datasets
import numpy as np
import random
import abc
import pickle
import tarfile
from datetime import datetime as dt
from tqdm import tqdm
import os

from . import clean_text

class DataSet(abc.ABC):
    '''
    Exposes interfaces to get the raw or cleaned data along with other metadata.

    The raw or clean data are returned along with the labels.

    Initially, before the model is trained, you need to call prepare_data() to
    split data into train and test sets.

    prepare_data() also cleans the split data and dumps to disk to make
    subsequent operations on cleaned data faster.

    TODO: After deployment, how to handle new documents. This can be a mono repo
    Or alternatively, the code that handles documents etc,
    can be a seperate repo altogether.

    Attributes:
    -----------
    _data_root: str = 'data'
        All data access will be constructed related to the data_root variable.
    _dataset_name: str = "scdb" or "others"
        Subdirectory for a specific dataset.

    By default data is expected to be in the path {data_root}/{dataset_name}.
    Train and test data should be at
        {data_root}/{_dataset_name}/train/
        {data_root}/{_dataset_name}/test/

    During initialization, the data will be cleaned, since this is a
    time consuming operation and saved at:
        {data_root}/{_dataset_name}/clean_train/
        {data_root}/{_dataset_name}/clean_test/

    Metadata related to the dataset should be kept at:
        {data_root}/{_dataset_name}/meta.csv
    This should be a tab seperated file. The labels corresponding to the
    docs should be in this file along with any other data.
    The train and test meta data will be seperated and saved to train_meta.csv
    and test_meta.csv

    Optional: Any other information about the dataset should be given at:
        {data_root}/{_dataset_name}/info.json

    This is the general approach to parsing the data. If any customization is
    needed for a particular dataset, any of the methods in this class can be
    overridden.

    TODO: best way to cache dataset, check other open sources like
    textacy, gensim, sklearn etc.

    '''

    def __init__(self, data_root: str = 'data', dataset_name: str = 'scdb') -> None:
        self._data_root = pathlib.Path(data_root)
        self._dataset_name = pathlib.Path(dataset_name)

        self._prefix = self._data_root / self._dataset_name
        self._corpus = self._prefix / 'corpus'

        self._train_files = self._prefix / 'train_files.pkl'
        self._test_files = self._prefix / 'test_files.pkl'

        self._clean_train_data_path = self._prefix / 'clean_train' # *.txt'
        self._clean_test_data_path = self._prefix / 'clean_test' # *.txt'

        self._train_meta_path = self._prefix / 'train_meta.csv'
        self._test_meta_path = self._prefix / 'test_meta.csv'

        self._meta = self._prefix / 'meta.csv'
        self._info = self._prefix / 'info.json'

    @abc.abstractmethod
    def prepare_data(self, overwrite:bool=False) -> None:
        '''
        Download, split and clean data.

        It first checks if dataset is in the relative path 'data'.
        If not, it tries to download from the cloud.
        If it cannot get from cloud, it raises an exception.

        Once the dataset is found in 'data/', it splits into train and test sets.
        The train and test splits are cleaned and put into seperate directories.

        TODO: implement download from cloud
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(self, clean: bool = False) -> Tuple[List[str], List[str], List[int], List[int]]:
        '''
        Get train and test sets.
        If clean == True, then it returns cleaned train and test sets with
        punctuation, Stop words, numbers removed and words stemed etc.
        Else, it returns the unprocessed data.

        Calls self.prepare_data() if dataset is not found in path.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_label_str(self, num :int) -> str:
        ''' Given a label number return its corresponding string. '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_label_dict(self) -> Dict:
        ''' Return the entire label number to string mapping. '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ''' Return meta of the train and test sets. '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_doc_metadata(self, doc_num:int=0, train:bool=True) -> pd.Series:
        ''' Return metadata of a particular document. '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_info(self) -> Dict:
        ''' Return any other info related to the dataset. '''
        raise NotImplementedError

class ScdbData(DataSet):
    '''
    ScdbData overrides DataSet class and provides data related to
    Supreme Court DataBase which can be got from here: http://scdb.wustl.edu/
    But, this data dump is taken from textacy library because of ease of access.

    To get a dump from textacy, call the method download_from_textacy().

    To prepare the train/test splits and clean the data, call prepare_data().
    This will internally call _download() to see if the processed data is
    available in cloud and download it. Else, it will split and clean the corpus
    in the local disk.

    Note that, there is only 1 document with label 14, so that is removed.
    '''

    # def __init__(self, data_root: str, dataset_name: str) -> None:
    def __init__(self, data_root: str = 'data', dataset_name: str = 'scdb') -> None:
        super().__init__(data_root, dataset_name)
        self.train_meta = None
        self.test_meta = None
        self.info = None
        self.issue_area_codes = None

    def download_from_textacy(self, overwrite: bool = False) -> None:
        '''
        Downloads the dataset from textacy and dumps to the relative path:
        data/scdb/.
        This is needed only once in the entire life cycle of the project.
        Once done, you can call prepare_data() to split data into train and test
        sets.
        '''

        if self._prefix.is_dir() \
            and self._corpus.is_dir() \
            and self._meta.is_file() \
            and self._info.is_file():
            if not overwrite:
                print(f"Dataset already preset. Not overwriting")
                return
            else:
                # May need to overwrite, if the textacy dataset is overwritten.
                print(f"Dataset present, overwriting")
        else:
            print(f"Dataset not present, downloading from textacy")
            # The dataset does not exist yet.
            self._corpus.mkdir(parents=True, exist_ok=True)

        ds = textacy.datasets.SupremeCourt()
        ds.download()
        print(f"dataset info:\n{ds.info}")
        ds = textacy.datasets.SupremeCourt()
        docs = []
        for i, record in enumerate(ds.records(limit=None)):
            text, meta = record
            if meta["issue_area"] == None or text == "":
                continue
            entry = {
                "issue": meta["issue"],
                "issue_area": meta["issue_area"],
                "decision_direction": meta["decision_direction"],
                "us_cite_id": meta["us_cite_id"],
                "maj_opinion_author": meta["maj_opinion_author"],
                "argument_date": meta["argument_date"],
                "decision_date": meta["decision_date"],
                "n_min_votes": meta["n_min_votes"],
                "n_maj_votes": meta["n_maj_votes"],
                "case_name": meta["case_name"],
                "text": text,
                }
            series = pd.Series(entry)
            docs.append(series)
        df = pd.DataFrame(data=docs)

        # delete the label and document with just one instance.
        df = df.drop((df.loc[df['issue_area'] == 14]).index)

        # Just to cross check
        check_list = random.sample(range(25), k=10)
        print(check_list)
        print(df.iloc[check_list]['issue_area'], df.iloc[check_list]['text'][:100])

        # Seperate out the text column from df.
        texts = df.pop('text')

        # 1. dump texts
        for i, text in enumerate(texts):
            file_name = self._corpus / (str(i).zfill(6) + '.txt')
            file_name.open('w').write(text)

        # 2. dump info
        scdb_meta_info = {
            'issue_area_codes': ds.issue_area_codes,
            'issue_codes': ds.issue_codes,
            'opinion_author_codes': ds.opinion_author_codes,
            'decision_directions': list(ds.decision_directions),
            'full_date_range': list(ds.full_date_range),
        }
        with open(self._info, "w") as outfile:
            json.dump(scdb_meta_info, outfile, indent=2)

        # 3. dump metadata
        values = {
                    'issue': -1,
                    'issue_area': -1,
                    'decision_direction': 'None',
                    'us_cite_id': 'None',
                    'maj_opinion_author': -1,
                    'argument_date': '1900-01-01',
                    'decision_date': '1900-01-01',
                    'n_min_votes': -1,
                    'n_maj_votes': -1,
                    'case_name': '',
                    }
        df.fillna(values, inplace=True)
        df.to_csv(self._meta, sep='|', index=False)

    def _download(self):
        ''' Download processed corpus from cloud '''
        from download_data_model import download
        download()

    def _split_data(self):
        '''
        Split data into train and test corpus. This will be a stratified split.
        That is, there will be similar distribution of documents in the train
        and test corpus wrt to the labels.
        '''
        print(f"Splitting data into train and test corpus")
        # read corpus
        corpus = sorted(self._corpus.glob('*.txt'))
        print(f"First and last few files: {corpus[:5]}, {corpus[-5:]}")

        # read metadata
        meta = pd.read_csv(self._meta, sep='|')
        values = {
                    'issue': -1,
                    'issue_area': -1,
                    'decision_direction': 'None',
                    'us_cite_id': 'None',
                    'maj_opinion_author': -1,
                    'argument_date': '1900-01-01',
                    'decision_date': '1900-01-01',
                    'n_min_votes': -1,
                    'n_maj_votes': -1,
                    'case_name': '',
                    }
        meta.fillna(values, inplace=True)
        meta = meta.astype({"issue": int}) # this is inferred as float, so.
        print(f"First and last few meta: {meta[:5]}, {meta[-5:]}")

        # train test split
        train_corpus, test_corpus, train_meta, test_meta = \
                            train_test_split(corpus, meta,
                                            test_size=0.2,
                                            random_state=42,
                                            stratify=meta[['issue_area']])
        print(f"{len(train_corpus)} train docs and {len(test_corpus)} test docs")

        # this is just for debugging. Actually will use pickle.
        with open(self._prefix / "train_files.txt", 'w') as fout:
            for c in train_corpus:
                fout.write(str(c) + '\n')
        with open(self._prefix / "test_files.txt", 'w') as fout:
            for c in test_corpus:
                fout.write(str(c) + '\n')

        # pickled file list
        with self._train_files.open('wb') as fout:
            pickle.dump(train_corpus, fout)
        with self._test_files.open('wb') as fout:
            pickle.dump(test_corpus, fout)

        # dump metadata
        train_meta.to_csv(self._train_meta_path, sep='|', index=False)
        test_meta.to_csv(self._test_meta_path, sep='|', index=False)

        # check if the split is proper.
        den = meta.groupby('issue_area')['case_name'].nunique().sum()
        total_groups = meta.groupby('issue_area')['case_name'].nunique().apply(lambda x: x/den*100)

        den = train_meta.groupby('issue_area')['case_name'].nunique().sum()
        train_groups = train_meta.groupby('issue_area')['case_name'].nunique().apply(lambda x: x/den*100)

        den = test_meta.groupby('issue_area')['case_name'].nunique().sum()
        test_groups = test_meta.groupby('issue_area')['case_name'].nunique().apply(lambda x: x/den*100)

        assert np.allclose(total_groups.to_numpy(), train_groups.to_numpy(), 0.1, 0.25)
        assert np.allclose(total_groups.to_numpy(), test_groups.to_numpy(), 0.1, 0.25)

    def _clean(self):
        ''' Cleans the corpus and saves to disk '''
        print(f"Cleaning data")
        with self._train_files.open('rb') as fin:
            train_files = pickle.load(fin)

        with self._test_files.open('rb') as fin:
            test_files = pickle.load(fin)

        train_corpus = clean_text.read_files(train_files)
        test_corpus = clean_text.read_files(test_files)

        print(f"Just a sanity check to see if raw corpus and clean corpus match")
        print(f"### Raw Train corpus ###")
        for text in train_corpus[:5]:
            print(text[:100])
            print('-'*10)
        print(f"### Raw Test corpus ###")
        for text in test_corpus[:5]:
            print(text[:100])
            print('-'*10)

        # clean texts
        print(f"Cleaning train corpus")
        start_time = dt.now()
        train_corpus = [clean_text.clean_text(text) for text in tqdm(train_corpus)]
        print(f"Cleaning test corpus")
        test_corpus = [clean_text.clean_text(text) for text in tqdm(test_corpus)]
        print(f"Cleaning corpus tooK {dt.now() - start_time}")

        print(f"### Clean Train corpus ###")
        for text in train_corpus[:5]:
            print(text[:100])
            print('-'*10)
        print(f"### Clean Test corpus ###")
        for text in test_corpus[:5]:
            print(text[:100])
            print('-'*10)

        pathlib.Path(self._clean_train_data_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self._clean_test_data_path).mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(train_corpus):
            out_name = self._clean_train_data_path / (str(i).zfill(6) + '.txt')
            out_name.open('w').write(text)
        for i, text in enumerate(test_corpus):
            out_name = self._clean_test_data_path / (str(i).zfill(6) + '.txt')
            out_name.open('w').write(text)

    def _check_data_sanity(self):
        '''
        Checks if the required files are present for working with this dataset.
        Else, returns false and the dataset can be reconstructed.
        '''
        if self._corpus.is_dir() \
            and self._train_files.is_file() \
            and self._test_files.is_file() \
            and self._clean_train_data_path.is_dir() \
            and self._clean_test_data_path.is_dir() \
            and self._train_meta_path.is_file() \
            and self._test_meta_path.is_file() \
            and self._meta.is_file() \
            and self._info.is_file():
            print(f"All files are present")
            corpus_count = len(list(self._corpus.glob('*')))
            clean_train_count = len(list(self._clean_train_data_path.glob('*')))
            clean_test_count = len(list(self._clean_test_data_path.glob('*')))
            if not corpus_count == (clean_train_count + clean_test_count):
                print(f"File count mismatch: corpus_count: {corpus_count}, " +
                    f"clean_train_count: {clean_train_count}, " +
                    f"cean_test_count: {clean_test_count}")
                return False
            return True
        else:
            print(f"Not all required files present")
            return False

    def prepare_data(self, overwrite=False) -> None:
        '''
        It will first call download to see if cleaned corpus is present in
        cloud and download it to data/scdb.
        Else, it will take the data obtained from textacy, from data/scdb and
        split it into train and test sets.
        These are then cleaned to remove stopwords, do stemming etc.
        The cleaned corpus can then be used to train models.
        '''
        print(f"Preparing SCDB data")
        if not self._check_data_sanity() or overwrite:
            print(f"Trying to download data from cloud")
            if not self._download(): # if not present in cloud.
                print(f"Data not found in cloud, downloading from textacy")
                self.download_from_textacy(overwrite=overwrite)
                self._split_data()
                self._clean()

    def get_data(self, clean: bool = False) -> \
                    Tuple[List[str], List[str], List[int], List[int]]:
        '''
        Returns the Train/Test corpus along with the labels.
        If clean is set to True, then the cleaned corpus is returned instead
        of the raw one.
        '''
        if clean:
            train_files = sorted(self._clean_train_data_path.glob('*'))
            test_files = sorted(self._clean_test_data_path.glob('*'))
        else:
            with self._train_files.open('rb') as fin:
                train_files = pickle.load(fin)

            with self._test_files.open('rb') as fin:
                test_files = pickle.load(fin)

        train_corpus = clean_text.read_files(train_files)
        test_corpus = clean_text.read_files(test_files)

        self.train_meta = pd.read_csv(self._train_meta_path, sep='|')
        self.test_meta = pd.read_csv(self._test_meta_path, sep='|')

        train_labels = self.train_meta['issue_area'].values
        test_labels = self.test_meta['issue_area'].values

        return train_corpus, test_corpus, train_labels, test_labels

    def get_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Returns the metadata of this dataset.
        This metadata contains the labels, issues, case name etc.
        '''
        # get train, test labels
        if self.train_meta is None:
            print("Loading train meta")
            self.train_meta = pd.read_csv(self._train_meta_path, sep='|')
        if self.test_meta is None:
            print("Loading test meta")
            self.test_meta = pd.read_csv(self._test_meta_path, sep='|')
        return self.train_meta, self.test_meta

    def get_doc_metadata(self, doc_num:int=0, train:bool=True) -> pd.Series:
        '''
        Return metadata of a particular document. By default, it returns the
        metadata from the training set, if train is set to False, it returns
        metadata of test set.
        '''
        if train:
            if self.train_meta is None:
                print("Loading train meta")
                self.train_meta = pd.read_csv(self._train_meta_path, sep='|')
            return self.train_meta.iloc[doc_num]
        else:
            if self.test_meta is None:
                print("Loading test meta")
                self.test_meta = pd.read_csv(self._test_meta_path, sep='|')
            return self.test_meta.iloc[doc_num]

    def get_info(self) -> Dict:
        '''
        Returns any information about the dataset.
        For SCDB, this contains, the mapping from label number to string,
        mapping from issues to issue description, mapping from author codes
        to author name, list of decision directions, start and end dates of
        court cases in this dataset.
        '''
        if not self.info:
            print(f"Loading info")
            with open(self._info, 'r') as fin:
                self.info = json.load(fin)
        return self.info

    def get_label_dict(self) -> Dict:
        ''' Returns the mapping from the label number to its string. '''
        if not self.info:
            print(f"Loading info")
            with open(self._info, 'r') as infile:
                self.info = json.load(infile)
        issue_area_codes = self.info['issue_area_codes']
        self.issue_area_codes = {int(k):v for k,v in issue_area_codes.items()}
        return self.issue_area_codes

    def get_label_str(self, num :int = -1) -> str:
        ''' Given a label number return its corresponding string. '''
        self.get_label_dict()
        return self.issue_area_codes[num]

class DataSetFactory:
    '''
    Factory which returns a DataSet object.
    Currently support SCDB data only.

    Methods
    -------
    get_data(name: str = 'scdb') -> DataSet
        Here name can be any of ['scdb']
    '''

    @staticmethod
    def get_data(root: str = 'data', name: str = 'scdb') -> DataSet:
        ''' Returns the concrete DataSet object '''
        if name == 'scdb':
            return ScdbData(root, name)
        else:
            raise Exception(f'Unsupported data type: {name}')
