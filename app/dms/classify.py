from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import random
from datetime import datetime as dt
import os.path
import pathlib

from utils.dataset import ScdbData

class Classify:

    def __init__(self, model_path='model/', ds_name="scdb"):
        if model_path and model_path != "":
            self.MODEL_PATH = model_path
        if ds_name and ds_name != "":
            self.DS_NAME = ds_name
        self.MODEL = None

    # TFIDF vectorizer
    def _build_vectorizer(self):
        # Vectorization parameters
        # Range (inclusive) of n-gram sizes for tokenizing text.
        NGRAM_RANGE = (1, 3)

        # Limit on the number of features. We use the top 20K features.
        TOP_K = 30000

        # Whether text should be split into word or character n-grams.
        # One of 'word', 'char'.
        TOKEN_MODE = 'word'

        # Minimum document/corpus frequency below which a token will be discarded.
        MIN_DOCUMENT_FREQUENCY = 2
        MAX_DOCUMENT_FREQUENCY = 0.9

        # Create keyword arguments to pass to the 'tf-idf' vectorizer.
        kwargs = {
                'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
                'lowercase': False,
                'analyzer': TOKEN_MODE,  # Split text into word tokens.
                'min_df': MIN_DOCUMENT_FREQUENCY,
                'max_df': MAX_DOCUMENT_FREQUENCY,
                'sublinear_tf':True,
                'max_features': TOP_K
        }
        vectorizer = TfidfVectorizer(**kwargs)
        return vectorizer

    def _build_classifier(self):
        sgdc = SGDClassifier(loss='hinge',
                        penalty='l2',
                        alpha=1e-5,
                        random_state=42,
                        max_iter=50,
                        tol=None)
        return sgdc

    def train(self, x_train, y_train):
        # get the data
        vect = self._build_vectorizer()
        clf = self._build_classifier()
        model = Pipeline([
            ('vect', vect),
            ('clf', clf)
            ])
        print(f"Starting Classifier training")
        start_time = dt.now()
        model.fit(x_train, y_train)
        pathlib.Path(self.MODEL_PATH).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.MODEL_PATH + self.DS_NAME + '_classifier.joblib')
        print(f"Training took {dt.now() - start_time}")

    def test(self, x_test, y_test):
        print(f"Starting Classifier testing")
        start_time = dt.now()
        model_path = self.MODEL_PATH + self.DS_NAME + '_classifier.joblib'
        if not os.path.isfile(model_path):
            print(f"No model found at path: {model_path}")
            return -1
        print(f"Loading model {self.DS_NAME} from path: {model_path}")
        model = joblib.load(model_path)
        y_pred = model.predict(x_test)
        print(f"Testing took {dt.now() - start_time}")
        print(classification_report(y_test, y_pred))

    def load(self):
        if not self.MODEL:
            model_path = self.MODEL_PATH + self.DS_NAME + '_classifier.joblib'
            if not os.path.isfile(model_path):
                print(f"No model found at path: {model_path}")
                return -1
            print(f"loading model {self.DS_NAME} from path: {model_path}")
            self.MODEL = joblib.load(model_path)
        return 0

    def predict(self, doc):
        if self.load() == -1: # load model
            print(f"No model found, exiting")
            return -1
        return self.MODEL.predict([doc])[0] # predict

if __name__ == "__main__":
    scdb = ScdbData()
    scdb.prepare_data()
    clean_train_corpus, clean_test_corpus, train_labels, test_labels = scdb.get_data(clean=True)
    print(f"len of train set: {len(train_labels)}")
    print(f"len of test set: {len(test_labels)}")

    # the classifier
    scdb_clf = Classify()

    # Train and Test
    print(f"Training the classifier")
    scdb_clf.train(clean_train_corpus, train_labels)
    print(f"Testing the classifier")
    scdb_clf.test(clean_test_corpus, test_labels)
