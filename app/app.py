from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Form
from typing import Optional, List, Literal
from fastapi.responses import PlainTextResponse, HTMLResponse
from pydantic import BaseSettings, BaseModel

from api.model import API
from api.types import ClfResponse, KpAlgosTypes
from api.types import SimTopics, SimTypes, SimResponse
from api.types import EntityLabels, EntityAttrs
from api.types import KpAlgos
from utils.filecloud import FileCloud
import json
from pathlib import Path


app = FastAPI()
filecloud = FileCloud()

class Config(BaseSettings):
    load_clf: bool = True
    load_sim_lda: bool = True
    load_sim_nmf: bool = True
    # sim_type: str = 'nmf'
    load_ner: bool = True
    load_kp: bool = True
    load_summ: bool = True
    load_search: bool = True

config = Config(_env_file='.config', _env_file_encoding='utf-8')
print(f"Features settings:")
print(config.dict())

class Api_Props(BaseModel):
    summary_count: int = 5
    keyphrase_count: int = 5
    sim_count: int = 5
    kp_algos: List[str] = ['cake']
    ner_labels: List[str] = ['ORG', 'GPE', 'PERSON', 'LAW']
    sim_types: List[str] = ['nmf']
    show_topics: bool = False

# if config.load_sim:
#     sim_type = SimTypes.lda if config.sim_type == "lda" else SimTypes.nmf
# else:
#     sim_type = None

api = API(
        clf=config.load_clf,
        sim_nmf=config.load_sim_nmf,
        sim_lda=config.load_sim_lda,
        ner=config.load_ner,
        kp=config.load_kp,
        summ=config.load_summ,
        search=config.load_search
    )

# TODO: pagination for similar documents.
def extract_features(text: str,
        corpus: List[str],
        api_props: Api_Props):
    # kp_algo_list: List[KpAlgosTypes] = [KpAlgosTypes.cake],
    # sim_count: int = 5,
    # sim_topics_enable: bool = False,
    # keyphrase_count: int = 5,
    # summary_count: int = 5,
    # ner_attrs: List[str] = ["text", "label_", "start", "end", "start_char", "end_char"],
    # ner_labels: List[str] = ["ORG", "GPE", "LAW", "LOC", "PERSON"]):
    a_cls_pred = ""
    a_sim_docs_lda = ""
    a_sim_topics_lda = ""
    a_sim_docs_nmf = ""
    a_sim_topics_nmf = ""
    a_entities = ""
    a_keyphrases = []
    a_summary = ""
    if config.load_clf:
        a_cls_pred = api.get_cls_label(doc=text)
    if config.load_sim_lda:
        a_sim_docs_lda = api.get_similar_documents_lda(doc=text, corpus=corpus,
                            count=api_props.sim_count, force=False)
        if api_props.show_topics: a_sim_topics_lda = api.get_topics_lda()
    if config.load_sim_nmf:
        a_sim_docs_nmf = api.get_similar_documents_nmf(doc=text, corpus=corpus,
                            count=api_props.sim_count, force=False)
        if api_props.show_topics: a_sim_topics_nmf = api.get_topics_nmf()
    if config.load_ner:
        # filtering of labels can be handled at UI layer
        # NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
        # NER_ATTRS = ["text", "label_", "start_char", "end_char"]
        # NER_LABELS = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW",
        #         "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON",
        #         "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
        # NER_LABELS = ["ORG", "GPE", "LAW", "LOC", "PERSON"]
        a_entities = api.get_entities(doc=text,
                                    filter_labels=api_props.ner_labels,
                                    )
    if config.load_kp:
        # TODO: proper api for algo input.
        # kp_algo_list = [
        #     KpAlgosTypes.cake,
        #     KpAlgosTypes.yake,
        #     KpAlgosTypes.textrank,
        #     KpAlgosTypes.pytextrank
        #     ]
        for kp_algo in api_props.kp_algos:
            kps = api.keyphrases(doc=text, algo=kp_algo,
                    TOP_N=api_props.keyphrase_count)
            a_keyphrases.append((kp_algo, kps))
    if config.load_summ:
        a_summary = api.summary(doc=text, TOP_N=api_props.summary_count)
    return {
            "cls": a_cls_pred,
            "sim_lda": a_sim_docs_lda,
            "sim_lda_topics": a_sim_topics_lda,
            "sim_nmf": a_sim_docs_nmf,
            "sim_nmf_topics": a_sim_topics_nmf,
            "entities": a_entities,
            "keyphrases": a_keyphrases,
            "summary": a_summary,
            "response": "OK"
            }


def extract_ner(text: str, labels: List[str] = None):
    return api.get_entities_html(text, filter_labels=labels)


def extract_highlights(text: str):
    return api.get_highlights_html(text)


# TODO: also return proper headers in responses to improve openapi documentation.

@app.get("/", response_class=PlainTextResponse)
async def root():
    ''' Welcome message '''
    return "Welcome to DMS Demo"


# TODO: handle pdf files
# @app.post("/doc")
# async def process_doc(file: UploadFile = File(...),
#                     prefix: str = Form(...),
#                     username: str = Form(...)):
#     # copy content
#     text = await file.read()
#     await file.seek(0)

#     # upload file to s3.
#     response = filecloud.upload_file(fileobj=file.file,
#                                     account=username,
#                                     prefix=prefix,
#                                     file_name=file.filename)
#     if response:
#         print(f"S3 file name: {response}")
#         # TODO: save this to db.
#     else:
#         print(f"Upload file failed")

#     # close file
#     await file.close()

#     # extract features
#     text = text.decode('utf-8', 'ignore')
#     return extract_features(text)


@app.post("/file-props")
async def file_props(
                    file_path: str = Form(...),
                    username: str = Form(...),
                    api_props: str = Form(...),
                    ):
    # download file from s3.
    text = filecloud.get_file(username, file_path)

    # check if sim model on disk
    # user_model_lda = Path('data/user_model/sim/lda.pkl')
    # user_model_nmf = Path('data/user_model/sim/nmf.pkl')

    # user_model_lda.mkdir(parents=True, exist_ok=True)
    # user_model_nmf.mkdir(parents=True, exist_ok=True)

    # if user_model_lda.exists():
    #     print('lda model exists')
    # else:
    #     print(f"creating lda model")
    # if user_model_nmf.exists():
    #     print('nmf model exists')
    # else:
    #     print(f"creating nmf model")

    # corpus = filecloud.get_all_files(account=username,
    #                                 max_items=20000)
    # corpus = corpus["file_list"]
    # corpus = [c.decode('utf-8', 'ignore') for c in corpus]

    """
    TODO
    on first user request:
        if sim model on disk (efs):
            no problem, just continue
        else if sim model is in s3:
            download the model from s3.
        else
            download users entire corpus.
            create the lda/nmf model.
            save the model back to s3.
            also cache in disk (efs). implement lru to delete (background tasks)

    * now the model is on disk (efs) *

    on upload new file:
        transform the corpus and update the model on disk (mark as dirty)

    on timeout:
        save the cached model back to s3 (background tasks)

    """

    # extract features
    if text:
        text = text.decode('utf-8', 'ignore')
        print(api_props)
        tmp = json.loads(api_props)
        print(tmp)
        api_props = Api_Props(**tmp)
        return extract_features(text, None, api_props)
    else:
        print(f"No such file for user {username}")
        return {"response": "NG"}


@app.post("/file-ner-html", response_class=HTMLResponse)
async def file_ner_html(
                    file_path: str = Form(...),
                    username: str = Form(...)):
    # upload file to s3.
    text = filecloud.get_file(username, file_path)

    # extract features
    if text:
        text = text.decode('utf-8', 'ignore')
        return extract_ner(text)
    else:
        # TODO: return HTML
        print(f"No such file for user {username}")
        return {"response": "NG"}


# TODO: handle pdf files
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...),
                    username: str = Form(...),
                    prefix: str = Form(None)):
    # upload file to s3.
    response = filecloud.upload_file(fileobj=file.file,
                                    account=username,
                                    prefix=prefix,
                                    file_name=file.filename)
    # close file
    await file.close()
    return response


# TODO: handle pdf files
@app.post("/upload-file-batch")
async def upload_file_batch(file_batch: List[UploadFile] = File(...),
                    username: str = Form(...),
                    prefix: str = Form(None)):
    print(type(file_batch))
    print(len(file_batch))
    for file in file_batch:
        print(type(file))
        print(file.filename)
        # upload file to s3.
        response = filecloud.upload_file(fileobj=file.file,
                                        account=username,
                                        prefix=prefix,
                                        file_name=file.filename)
        # await file.seek(0)
        # text = await file.read()
        # index file in es

        # index file in similarity - lda

        # index file in similarity - nmf

        # get file ners

        # get file keywords

        # get file highlights

        # get file class

        # close file
        await file.close()
        print(f"file: {file.filename} - upload status: {response}")
    return response


@app.post("/list-files")
async def list_files(username: str = Form(...),
                    prefix: Optional[str] = Form(None),
                    max_items: Optional[str] = Form(None),
                    next_token: Optional[str] = Form(None)):
    return filecloud.list_files(username, prefix, max_items, next_token)
    # return filecloud.list_files_page(username, prefix, next_token)


@app.post("/list-file-versions")
async def list_file_versions(username: str = Form(...),
                    file_path: str = Form(...)):
    return filecloud.list_file_versions(username, file_path)


@app.post("/get-file", response_class=PlainTextResponse)
async def get_file(username: str = Form(...),
                    file_path: str = Form(...),
                    version_id: Optional[str] = Form(None)):
    return filecloud.get_file(username, file_path, version_id)


@app.post("/get-file-highlights", response_class=HTMLResponse)
async def get_file_highlights(username: str = Form(...),
                    file_path: str = Form(...),
                    version_id: Optional[str] = Form(None)):
    text = filecloud.get_file(username, file_path, version_id)
    if text:
        text = text.decode('utf-8', 'ignore')
        return extract_highlights(text)
    else:
        # TODO: return html
        print(f"No such file for user {username}")
        return {"response": "NG"}


# TODO: also return Content-Length, Last-Modified and ETag in headers
@app.post("/download-file")
def download_file(username: str = Form(...),
                    file_path: str = Form(...),
                    version_id: Optional[str] = Form(None)):
    return filecloud.download_file(username, file_path, version_id)


@app.post("/delete-file")
async def delete_file(username: str = Form(...),
                    file_path: str = Form(...),
                    version_id: Optional[str] = Form(None)):
    return filecloud.delete_file(username, file_path, version_id)


@app.get("/get-api-props")
async def get_api_props():
    ner_labels, ner_attrs = api.get_entity_labels_attrs()
    kp_algos = api.get_keyphrase_algos()
    sim_types = api.get_sim_types()
    return {
        "ner_labels": ner_labels,
        "ner_attrs": ner_attrs,
        "kp_algos": kp_algos,
        "sim_types": sim_types
    }



# g_api_props = Api_Props()

# @app.post("/set-api-props")
# async def set_api_props(api_props: Api_Props):
#     global g_api_props
#     g_api_props = api_props
#     print(g_api_props.dict())


# TODO: handle pdf files
if config.load_search:
    # search
    @app.get("/search/{query_text}")
    def query(query_text: str) -> List[str]:
        '''
        Search for a text in the corpus
        '''
        return api.query(query_text=query_text)

    @app.get("/search-history")
    def search_history() -> List[str]:
        '''
        Returns the Search history
        '''
        return api.get_search_history()
