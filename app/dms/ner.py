from wasabi import table
import spacy
from spacy import displacy
import random


class Ner:

    NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
    NER_LABELS = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW",
                "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON",
                "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]

    def __init__(self, nlp=None):
        print(f"NER init")
        if not nlp:
            print(f"Loading spacy model")
            self.nlp = spacy.load("en_core_web_md")
        else:
            self.nlp = nlp
        # self.NER_LABELS = self.nlp.get_pipe("ner").labels

    def get_named_entities(self, text, filter_labels=None,
                            filter_attrs=['text', 'label_']):
        # This is one area with a lot of scope of improvement.
        # TODO: Train a legal domain specific model.

        if text == None or text == "":
            print("No text input provided")
            return None

        doc = self.nlp(text)
        if not filter_labels:
            filter_labels = self.NER_LABELS

        ners = [
            [str(getattr(ent, attr)) for attr in filter_attrs]
            for ent in doc.ents
            if ent.label_ in filter_labels
        ]

        return ners

    def get_named_entities_html(self, text, filter_labels=None):
        # This is one area with a lot of scope of improvement.
        # TODO: Train a legal domain specific model.

        if text == None or text == "":
            print("No text input provided")
            return None

        doc = self.nlp(text)
        if not filter_labels:
            filter_labels = self.NER_LABELS
        options = {"ents": filter_labels}
        html = displacy.render(doc, style="ent", options=options, page=True)

        return html

    def get_entity_labels(self):
        return self.NER_LABELS

    def get_entity_attrs(self):
        return self.NER_ATTRS

    def serve_named_entities(self, text):
        # TODO: Displacy HTML render.

        if text == None or text == "":
            print("No text input provided")
            return None

        nlp = spacy.load("en_core_web_md")
        doc = nlp(text)
        html = displacy.render(doc, style="ent", page=True)
        with open('ner.html', 'w') as fout:
            fout.write(html)
        html = displacy.render(doc, style="ent", minify=True)
        with open('ner_minify.html', 'w') as fout:
            fout.write(html)
