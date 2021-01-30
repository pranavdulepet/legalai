from pydantic import BaseModel, ValidationError, validator
from typing import List, Tuple
from enum import Enum

from dms.ner import Ner

class ClfResponse(BaseModel):
    pred_label_num: int
    pred_label_str: str

class SimTopics(BaseModel):
    topics: List[str]

class SimDocAttr(BaseModel):
    doc_num: int
    text: str
    actual_label_num: int
    actual_label_str: str
    doc_topics: List[Tuple[int, float]]

class SimDocs(BaseModel):
    doc: SimDocAttr
    score: float

class SimResponse(BaseModel):
    doc: List[Tuple[int, float]]
    sim_docs_list: List[SimDocs]

class SimTypes(str, Enum):
    lda = 'lda'
    nmf = 'nmf'

class EntityLabels(BaseModel):
    ent_labels: List[str] = Ner.NER_LABELS

    @validator('ent_labels')
    def correct_labels(cls, v):
        if not all(l in Ner.NER_LABELS for l in v):
            print(f'{v} not valid')
            raise ValueError('Incorrect labels')
        return v

class EntityAttrs(BaseModel):
    ent_attrs: List[str] = Ner.NER_ATTRS

    @validator('ent_attrs')
    def correct_attrs(cls, v):
        if not all(l in Ner.NER_ATTRS for l in v):
            print(f'{v} not valid')
            raise ValueError('Incorrect attributes')
        return v

class KpAlgosTypes(str, Enum):
    cake = 'cake'
    yake = 'yake'
    textrank = 'textrank'
    pytextrank = 'pytextrank'

class KpAlgos(BaseModel):
    algo: KpAlgosTypes = KpAlgosTypes.cake
