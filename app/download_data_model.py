#! /usr/bin/env python

import tarfile
from datetime import datetime as dt
import boto3
import os
from dotenv import load_dotenv

# TODO: make this class generic.
class CloudStorage:
    @classmethod
    def get_files(cls) -> None:
        ''' Downloads the model and data from hosted Cloud Service '''
        load_dotenv('.config')
        CLOUD_PROVIDER = os.getenv('CLOUD_STORAGE_PROVIDER')
        if CLOUD_PROVIDER == "AWS":
            return cls.get_from_aws()
        else:
            raise Exception(f'Unsupported cloud provider: {CLOUD_PROVIDER}')

    @classmethod
    def get_from_aws(cls):
        start_time = dt.now()
        # which region
        AWS_REGION = os.getenv('AWS_REGION')

        # get the bucket and file names
        ssm = boto3.client('ssm', region_name=AWS_REGION)
        AWS_S3_BUCKET = ssm.get_parameter(Name="/docmansys/prod/bucket")['Parameter']["Value"]
        AWS_S3_DATA_FILE = ssm.get_parameter(Name="/docmansys/prod/data")['Parameter']["Value"]
        AWS_S3_MODEL_FILE = ssm.get_parameter(Name="/docmansys/prod/model")['Parameter']["Value"]

        # get the files
        s3 = boto3.client('s3')
        s3.download_file(AWS_S3_BUCKET, AWS_S3_DATA_FILE, 'data.tar.gz')
        s3.download_file(AWS_S3_BUCKET, AWS_S3_MODEL_FILE, 'model.tar.gz')

        # extract them
        with tarfile.open('data.tar.gz') as tfile:
            tfile.extractall('./')
        with tarfile.open('model.tar.gz') as tfile:
            tfile.extractall('./')

        # delete the compressed ones.
        os.remove('data.tar.gz')
        os.remove('model.tar.gz')

        print(f"Time to download and extract: {dt.now()-start_time}")


def download():
    ''' Download processed corpus from cloud '''
    objects = CloudStorage.get_files()


if __name__ == "__main__":
    download()
